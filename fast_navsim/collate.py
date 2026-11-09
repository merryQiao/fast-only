"""One current or four chronological original images with NAVSIM timestamps."""
import random
import threading
import torch
from PIL import ImageEnhance
from fast_only.common import prepare_original_image

HISTORY_TIMES=(-1.5,-1.,-.5,0.)


def _history_times(history_times_s):
    try:times=tuple(history_times_s)
    except TypeError as error:raise ValueError('Unsupported NAVSIM time policy') from error
    if any(isinstance(t,bool) for t in times) or times not in (HISTORY_TIMES,(0.,)):
        raise ValueError('Unsupported NAVSIM time policy')
    return times


def temporal_content(frames,history_times_s=HISTORY_TIMES):
    times=_history_times(history_times_s)
    if len(frames)!=len(times):raise ValueError(f'Expected {len(times)} chronological NAVSIM frames')
    content=[]
    for t,image in zip(times,frames):
        content.extend([{'type':'text','text':f'Front camera t{t:+.1f}s: '},
                        {'type':'image','image':image}])
    content.append({'type':'text','text':'Plan eight future ego poses (x, y, heading) at 0.5 second intervals over 4 seconds.'})
    return content


class NavsimCollator:
    def __init__(self,model_path,*,policy='original_pad',pad_multiple=32,pad_fill=(0,0,0),
                 history_times_s=HISTORY_TIMES,prompt_template='navsim_temporal_v1',brightness=0.):
        from transformers import AutoProcessor
        times=_history_times(history_times_s)
        if policy!='original_pad' or prompt_template!='navsim_temporal_v1':
            raise ValueError('Unsupported NAVSIM processor/time policy')
        self.history_times_s=times
        self.image_frames=len(times)
        self.processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
        self.processor.tokenizer.padding_side='left'
        image_processor=self.processor.image_processor
        if pad_multiple!=image_processor.patch_size*image_processor.merge_size:
            raise ValueError('Padding must equal native patch*merge')
        image_processor.do_resize=False
        self.pad_multiple=pad_multiple;self.pad_fill=pad_fill;self.brightness=brightness
        self._processor_lock=threading.RLock()

    def __call__(self,samples,seed=None):
        rng=random.Random(seed) if seed is not None else random
        contents=[];images=[];audits=[]
        for sample in samples:
            frames=sample['images']
            if len(frames)!=self.image_frames:raise ValueError(f'Expected {self.image_frames} source images')
            if self.brightness:
                factor=rng.uniform(1-self.brightness,1+self.brightness)
                frames=[ImageEnhance.Brightness(image).enhance(factor) for image in frames]
            prepared=[prepare_original_image(image,self.pad_multiple,self.pad_fill) for image in frames]
            frames=[p[0] for p in prepared];audits.extend(p[1] for p in prepared)
            contents.append(temporal_content(frames,self.history_times_s));images.extend(frames)
        with self._processor_lock:
            prompts=[self.processor.apply_chat_template([{'role':'user','content':c}],tokenize=False,
                     add_generation_prompt=False) for c in contents]
            native=dict(self.processor(text=prompts,images=images,padding=True,return_tensors='pt',
                        images_kwargs={'do_resize':False}))
        grids=native['image_grid_thw'].tolist()
        if len(grids)!=len(audits):raise RuntimeError('Image count mismatch')
        patch=self.processor.image_processor.patch_size
        merge=self.processor.image_processor.merge_size
        for audit,grid in zip(audits,grids):
            if audit['processed_size_wh']!=[grid[2]*patch,grid[1]*patch]:
                raise RuntimeError('Native processor resized original input')
            audit['image_grid_thw']=grid;audit['visual_tokens']=grid[0]*grid[1]*grid[2]//merge**2
        return {'native':native,'image_audit':audits,
                # Shared benchmark forwards this sentinel; the model creates
                # no target feature/token and rejects any non-None value.
                'nav':None,**{k:torch.stack([s[k] for s in samples])
                for k in ['ego','command','trajectory_gt']}}
