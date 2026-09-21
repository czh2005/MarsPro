"""Batch-local exact sliding-window implementation; deployment files untouched."""
import torch
import torch.nn.functional as F
from transformers.models.esmfold2 import modeling_esmfold2_common as c
def blocked_forward(self,x,attention_params):
    B,N=x.shape[:2];cos,sin=attention_params[:2]
    qkv=self.Wqkv(x).view(B,N,3,self.n_heads,self.head_dim).permute(2,0,1,3,4)
    q,k,v=qkv.unbind(0)
    q=c.apply_rotary_emb_3d(c.qk_norm(q),cos,sin).transpose(1,2)
    k=c.apply_rotary_emb_3d(c.qk_norm(k),cos,sin).transpose(1,2)
    v=v.transpose(1,2)
    out=torch.empty_like(q)
    # The batch contains exactly one unpadded canonical protein.
    for start in range(0,N,256):
        end=min(start+256,N);lo=max(0,start-self.half_window);hi=min(N,end+self.half_window)
        score=torch.matmul(q[:,:,start:end],k[:,:,lo:hi].transpose(-2,-1))*self.scale
        mask=(torch.arange(start,end,device=x.device)[:,None]-torch.arange(lo,hi,device=x.device)[None,:]).abs()>self.half_window
        score=score.masked_fill(mask,float("-inf"))
        out[:,:,start:end]=torch.matmul(F.softmax(score,dim=-1),v[:,:,lo:hi])
    out=out.transpose(1,2).reshape(B,N,-1)*torch.sigmoid(self.gate_proj(x))
    return self.out_proj(out)
def install():
    torch.manual_seed(19)
    module=c.SWA3DRoPEAttention(64,4,half_window=64).eval()
    maximum=0.0
    with torch.inference_mode():
        for n in [1,63,129,257,601,1021]:
            x=torch.randn(1,n,64);angle=torch.randn(1,n,8)
            params=(angle.cos(),angle.sin())
            expected=module(x,params);actual=blocked_forward(module,x,params)
            maximum=max(maximum,(expected-actual).abs().max().item())
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    c.SWA3DRoPEAttention.forward=blocked_forward
    return dict(status="pass",lengths=[1,63,129,257,601,1021],max_absolute_error=maximum,reference="Dense masked FP32 attention",query_block=256)



