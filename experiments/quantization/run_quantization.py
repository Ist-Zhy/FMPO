"""FMPO post-training quantization: prepare and build TensorRT engines.

Uses strict checkpoint loading, merges LoRA, folds weight standardization,
and calibrates signed INT8 convolution/linear inputs on TRAIN images only.
Norms, attention arithmetic, and residual arithmetic remain floating point.
"""
import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from Model.Flow_MMIF import FlowMatching, UnetVelocity, WeightStandardizedConv2d

CHECKPOINTS = {
    64: Path(os.environ.get('FMPO_CHECKPOINT_64', ROOT / 'weights/fmpo_dim64.pt')),
    32: Path(os.environ.get('FMPO_CHECKPOINT_32', ROOT / 'weights/fmpo_dim32.pt')),
    16: Path(os.environ.get('FMPO_CHECKPOINT_16', ROOT / 'weights/fmpo_dim16.pt')),
}


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_map(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    def natural_key(path):
        return [int(s) if s.isdigit() else s.lower()
                for s in re.split(r"(\d+)", path.stem)]
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return {p.stem: p for p in sorted(files, key=natural_key)}


def test_pairs_if_configured(dataset):
    """Optionally check calibration/test overlap when test paths are available."""
    paths = [Path(os.environ.get(f'FMPO_{dataset}_{kind.upper()}',
                  ROOT / 'data' / dataset / 'test' / kind)) for kind in ('ir', 'vi')]
    configured = any(f'FMPO_{dataset}_{kind.upper()}' in os.environ for kind in ('ir', 'vi'))
    if configured or any(path.exists() for path in paths):
        return pairs(*paths)
    return []


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def pairs(ir, vi):
    a, b = image_map(ir), image_map(vi)
    matched = [(k, a[k], b[k]) for k in a if k in b]
    if not matched:
        raise ValueError(f'No matched image pairs in {ir} and {vi}')
    return matched


def select(items, count):
    return [items[i] for i in np.linspace(0, len(items)-1, min(count, len(items)), dtype=int)]


def read_pair(pair, device='cuda', size=None):
    out = []
    for path in pair[1:]:
        im = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        if size:
            im = cv2.resize(im, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        out.append(torch.from_numpy(im.copy()).permute(2,0,1).unsqueeze(0).to(device).float().contiguous()/255)
    if out[0].shape != out[1].shape:
        raise ValueError(f'Unaligned source shapes: {pair[0]}')
    return out


def load_flow(dim, device):
    flow = FlowMatching(UnetVelocity(dim=dim, dim_mults=(1,1,1,1), condition=True),
                        image_size=256, sampling_timesteps=1, condition=True,
                        sigma_min=0., sigma_max=0.).to(device)
    state = torch.load(CHECKPOINTS[dim], map_location=device, weights_only=False)['model']
    has_lora = any('lora_A' in k for k in state)
    if has_lora:
        from peft import LoraConfig, get_peft_model
        flow.model = get_peft_model(flow.model, LoraConfig(r=16, lora_alpha=32,
            target_modules=['to_qkv', 'to_out.0'], lora_dropout=0.05, bias='none'))
    flow.load_state_dict(state, strict=True)
    return flow.eval(), has_lora


class OneStep(nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = velocity
    def forward(self, ir, vi):
        ir, vi = ir*2-1, vi*2-1
        source = (ir+vi)*0.5
        t = torch.zeros(ir.shape[0], dtype=ir.dtype, device=ir.device)
        return ((source + self.velocity(torch.cat([source,ir,vi],1), t))+1)*0.5


def fold_ws(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, WeightStandardizedConv2d):
            new = nn.Conv2d(child.in_channels, child.out_channels, child.kernel_size,
                child.stride, child.padding, child.dilation, child.groups,
                bias=child.bias is not None).to(child.weight.device)
            with torch.no_grad():
                w = child.weight
                dims = tuple(range(1,w.ndim))
                new.weight.copy_((w-w.mean(dims,keepdim=True))*
                    (w.var(dims,unbiased=False,keepdim=True)+1e-5).rsqrt())
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(module, name, new)
            count += 1
        else:
            count += fold_ws(child)
    return count


class QuantizedOp(nn.Module):
    """Q/DQ simulation for export; TensorRT executes supported ops in INT8."""
    def __init__(self, op, input_max):
        super().__init__()
        self.op = op
        self.register_buffer('activation_scale', torch.tensor(max(input_max/127,1e-8), device=op.weight.device))
        self.register_buffer('activation_zero',torch.tensor(0,device=op.weight.device,dtype=torch.int32))
        dims = tuple(range(1,op.weight.ndim))
        self.register_buffer('weight_scale',op.weight.detach().abs().amax(dims).clamp_min(1e-8)/127)
        self.register_buffer('weight_zero',torch.zeros(op.weight.shape[0],device=op.weight.device,dtype=torch.int32))
    def forward(self, x):
        x = torch.fake_quantize_per_tensor_affine(x,self.activation_scale,self.activation_zero,-128,127)
        w = torch.fake_quantize_per_channel_affine(self.op.weight,self.weight_scale,self.weight_zero,0,-128,127)
        if isinstance(self.op,nn.Conv2d):
            return F.conv2d(x,w,self.op.bias,self.op.stride,self.op.padding,self.op.dilation,self.op.groups)
        return F.linear(x,w,self.op.bias)


def quantize(module, ranges, prefix=''):
    for name, child in list(module.named_children()):
        key = f'{prefix}.{name}' if prefix else name
        if isinstance(child,(nn.Conv2d,nn.Linear)):
            setattr(module,name,QuantizedOp(child,ranges[key]))
        else:
            quantize(child,ranges,key)


def export(model, path, inputs):
    import onnx
    torch.onnx.export(model, tuple(inputs), str(path), opset_version=17,
        input_names=['ir','vi'],output_names=['fused'],
        dynamic_axes={name:{2:'height',3:'width'} for name in ['ir','vi','fused']},
        do_constant_folding=True)
    onnx.checker.check_model(str(path))


def prepare(args, out):
    flow,lora = load_flow(args.dim,'cuda')
    folder = Path(os.environ.get('FMPO_MSRS_CALIBRATION', ROOT / 'data' / 'MSRS' / 'train'))
    data = pairs(folder / 'ir', folder / 'vi')
    inp=read_pair(data[0],size=(256,256))
    wrapper=OneStep(flow.model).eval()
    with torch.no_grad():
        reference=flow.sample(x_input=inp,batch_size=1,last=True)
        step_error=(wrapper(*inp)-reference).abs().max().item()
        if lora:
            flow.model=flow.model.merge_and_unload()
            wrapper=OneStep(flow.model).eval()
        merge_error=(wrapper(*inp)-reference).abs().max().item()
        folded_count=fold_ws(wrapper)
        fold_error=(wrapper(*inp)-reference).abs().max().item()
    if max(step_error,merge_error,fold_error) >= 1e-4:
        raise RuntimeError(f'Model conversion parity failed: {(step_error, merge_error, fold_error)}')
    torch.save(wrapper.state_dict(),out/'folded_fp32.pt')
    ranges={}
    handles=[]
    def observer(name):
        def hook(mod, inputs):
            ranges[name]=max(ranges.get(name,0.),float(inputs[0].detach().abs().max()))
        return hook
    for name,mod in wrapper.named_modules():
        if isinstance(mod,(nn.Conv2d,nn.Linear)):
            handles.append(mod.register_forward_pre_hook(observer(name)))
    calibration=[]
    with torch.no_grad():
        for ds in ['MSRS','FMB']:
            folder=Path(os.environ.get(f'FMPO_{ds}_CALIBRATION', ROOT / 'data' / ds / 'train'))
            chosen=select(pairs(folder/'ir',folder/'vi'),args.calibration//2)
            test=test_pairs_if_configured(ds)
            if not test:
                print(f'{ds}: no test paths configured; calibration/test overlap was not checked', flush=True)
            # Reject identical image pairs across calibration and test even if filenames differ.
            def digest(pair):
                return hashlib.sha256(pair[1].read_bytes()+pair[2].read_bytes()).hexdigest()
            test_hashes={digest(p) for p in test}
            for pair in chosen:
                if digest(pair) in test_hashes:
                    raise ValueError(f'Calibration/test overlap: {pair[0]}')
                wrapper(*read_pair(pair,size=(480,640)))
                calibration.append({'dataset':ds,'ir':str(pair[1]),'vi':str(pair[2]),'sha256':digest(pair)})
                print(f'calibration {len(calibration)}/{args.calibration}',flush=True)
    for h in handles:h.remove()
    save_json(out/'calibration.json',{'images':calibration,'input_absmax':ranges,'size':[480,640]})
    export(wrapper,out/'fp32.onnx',inp)
    quantized=copy.deepcopy(wrapper)
    quantize(quantized,ranges)
    export(quantized,out/'int8_qdq.onnx',inp)
    metadata={'dim':args.dim,'checkpoint':str(CHECKPOINTS[args.dim]),
        'checkpoint_sha256':hashlib.sha256(CHECKPOINTS[args.dim].read_bytes()).hexdigest(),
        'strict_load':True,'lora':lora,'lora_r':16 if lora else None,'lora_alpha':32 if lora else None,
        'one_step_max_error':step_error,'lora_merge_max_error':merge_error,'ws_fold_max_error':fold_error,
        'ws_layers_folded':folded_count,'quantized_ops':len(ranges),
        'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'quantization':'Per-output-channel symmetric INT8 weights; per-tensor symmetric minmax INT8 inputs for Conv2d/Linear; other arithmetic floating point',
        'calibration_count':len(calibration)}
    save_json(out/'metadata.json',metadata)
    print(metadata,flush=True)


def build(args,out):
    import tensorrt as trt
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    path=out/('int8_qdq.onnx' if args.mode=='int8' else 'fp32.onnx')
    if not parser.parse(path.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,2<<30)
    config.builder_optimization_level=2
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    if args.mode in ['fp16','int8']:config.set_flag(trt.BuilderFlag.FP16)
    profile=builder.create_optimization_profile()
    for name in ['ir','vi']:
        profile.set_shape(name,(1,3,256,256),(1,3,480,640),(1,3,600,800))
    config.add_optimization_profile(profile)
    started=time.time()
    engine=builder.build_serialized_network(network,config)
    if engine is None:raise RuntimeError('TensorRT engine build failed')
    (out/f'{args.mode}.engine').write_bytes(bytes(engine))
    print(f'Built {args.mode} in {time.time()-started:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','build'])
    parser.add_argument('--dim',type=int,default=64,choices=[16,32,64])
    parser.add_argument('--mode',choices=['fp32','fp16','int8'],default='fp32')
    parser.add_argument('--calibration',type=int,default=32)
    args=parser.parse_args()
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'results_quantization'/f'dim{args.dim}';out.mkdir(parents=True,exist_ok=True)
    if args.calibration < 2:
        parser.error('--calibration must be at least 2')
    {'prepare':prepare,'build':build}[args.action](args,out)
if __name__=='__main__':main()
