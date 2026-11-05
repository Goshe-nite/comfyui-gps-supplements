import re

import comfy.samplers


def is_link(value):
    """Return True if value is an API prompt link reference, [node_id, slot]."""
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


def get_node(prompt, unique_id):
    """Return the prompt entry for a node. Raise ValueError if missing."""
    node = prompt.get(unique_id)
    if node is None:
        raise ValueError(f"Node {unique_id} not found in execution prompt")
    return node


def get_linked_source(prompt, unique_id, input_name):
    """Return the source node id of a linked input.

    The execution prompt (API format) stores linked inputs as
    [source_node_id, output_slot] pairs. Source ids may be subgraph path
    ids like "60:56". The source node is always present in the prompt,
    because it executes before the consumer.
    """
    node = get_node(prompt, unique_id)
    value = node.get("inputs", {}).get(input_name)
    if not is_link(value):
        raise ValueError(f"Input '{input_name}' of node {unique_id} is not linked")
    return value[0]


def resolve_value(value, prompt, depth=0):
    """Resolve a prompt input value to a plain value.

    Follow link references to the source node. For a linked value, read
    the source node input at the slot index. This returns the value for
    pass-through nodes (for example Basic Seed). Return None if the
    value is not resolvable from the prompt.
    """
    if depth >= 100:
        raise ValueError("Over 99 nodes searched, consider connecting the root node directly instead.")
    if not is_link(value):
        return value
    node_id, slot = value[0], value[1]
    node = prompt.get(node_id)
    if node is None:
        return None
    inputs = list(node.get("inputs", {}).values())
    if slot >= len(inputs):
        return None
    return resolve_value(inputs[slot], prompt, depth + 1)


def find_model_name(prompt, node_id, depth=0):
    """Walk a model chain and return the loader model name.

    Start at node_id. Return the first *_name widget value found.
    Follow the model input to the previous node in the chain. Return
    None if no name is found.
    """
    if depth >= 10:
        return None
    node = prompt.get(node_id)
    if node is None:
        return None
    inputs = node.get("inputs", {})
    for key, value in inputs.items():
        # Skip sampler_name: it is a sampler setting, not a model name.
        if key == "sampler_name":
            continue
        if re.search(".*_name.*", key) and isinstance(value, str):
            return value
    model_input = inputs.get("model")
    if is_link(model_input):
        return find_model_name(prompt, model_input[0], depth + 1)
    return None


def round2(value):
    """Round a number to 2 decimals. Pass through non-numbers."""
    return round(value, 2) if isinstance(value, (int, float)) and not isinstance(value, bool) else value


class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False
any = AnyType("*")

class modelToString:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                         "model": (any, {}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT",
                       "unique_id": "UNIQUE_ID",},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_model"
    CATEGORY = "GPSupps"
    DESCRIPTION = """
Connect to model node to turn the model name (with the partial path if exist) to a string, detects model name by widget name 
*_name. Walks the model chain back to the loader, so intermediate 
nodes (for example ModelSamplingFlux) do not hide the name. 
"""

    def get_model(self, extra_pnginfo, prompt, unique_id, model=None):
        source_id = get_linked_source(prompt, unique_id, "model")
        # Walk the model chain so intermediate nodes (for example
        # ModelSamplingFlux) do not hide the loader name.
        model_name = find_model_name(prompt, source_id)

        return (model_name,)

class loraToString:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                         "lora_loader": (any, {}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT",
                       "unique_id": "UNIQUE_ID",},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_rgthree_loras"
    CATEGORY = "GPSupps"
    DESCRIPTION = """
Connect to power lora loader (rgthree) to output a string list of loras with their strengths,
in the format <lora1:strength><lora2:strength>.... 
"""

    def get_rgthree_loras(self, extra_pnginfo, prompt, unique_id, lora_loader=None):
        source_id = get_linked_source(prompt, unique_id, "lora_loader")
        source = prompt[source_id]
        if source["class_type"] != "Power Lora Loader (rgthree)":
            raise ValueError("Please check if Power Lora Loader (rgthree) is set as input.")

        loras = []
        for key, value in source["inputs"].items():
            if re.search("^lora_.*", key) and isinstance(value, dict) and value.get('on'):
                loras.append(f"<lora:{value['lora']}:{value['strength']}>")

        return ("".join(loras),)
    
class loraPromptConcat:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                         "lora_loader": (any, {}),
                         "positive":("STRING", {"default": '', "forceInput": True}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT",
                       "unique_id": "UNIQUE_ID",},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "conditioning_to_string"
    CATEGORY = "GPSupps"
    DESCRIPTION = """
Connect to a power lora loader (rgthree) and text positive prompt to output a concatenated string 
of the prompt and loras with their strengths (in the format <lora:strength>), designed for comfyui-image-saver, for 
adding lora information to civitai compatible images. 
"""

    def conditioning_to_string(self, extra_pnginfo, prompt, unique_id, positive, lora_loader=None):
        source_id = get_linked_source(prompt, unique_id, "lora_loader")
        source = prompt[source_id]
        if source["class_type"] != "Power Lora Loader (rgthree)":
            raise ValueError("Please check if Power Lora Loader (rgthree) is set as input.")

        loras = []
        for key, value in source["inputs"].items():
            if re.search("^lora_.*", key) and isinstance(value, dict) and value.get('on'):
                loras.append(f"<lora:{value['lora']}:{value['strength']}>")

        return (positive+"".join(loras),)
    
class ksamplerToImageSaver:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                         "ksampler": (any, {}),
                         "seed_value": ("INT", {
                            "default": 0,
                            "min": 0,
                            "step": 1,
                            }),
                         "steps": ("INT", {
                            "default": 30,
                            "min": 1,
                            "step": 1,
                            }),
                         "cfg": ("FLOAT", {
                            "default": 8.0,
                            "min": 0.0,
                            "step": 0.5,
                            }),
                         "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                         "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                         "denoise": ("FLOAT", {
                            "default": 1.0,
                            "min": 0.0,
                            "max": 1.0,
                            "step": 0.05,
                            }),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT",
                       "unique_id": "UNIQUE_ID",},
        }

    RETURN_TYPES = ("INT","INT", "FLOAT", "STRING","STRING","FLOAT","STRING",)
    RETURN_NAMES = ("seed_value","steps", "cfg","sampler_name","scheduler","denoise","model_name",)
    FUNCTION = "get_ksampler_config"
    CATEGORY = "GPSupps"
    DESCRIPTION = """
Connect to any output of KSampler and alike to retrieve seed_value, steps, cfg, sampler_name, 
scheduler, denoise, model_name and output the values in image saver(ComfyUI-Image-Saver) compatible format. 
model_name walks the model chain back to the loader and reads the *_name widget.
Input should work with KSampler Config (rgthree). Fallback values 
can be entered for use with low compatibility nodes. 
"""

    def get_ksampler_config(self, extra_pnginfo, prompt, unique_id,seed_value,steps,cfg, sampler_name, scheduler,denoise, ksampler=None,):
        try:
            source_id = get_linked_source(prompt, unique_id, "ksampler")
        except ValueError:
            # Input not linked: return the fallback widget values.
            return (seed_value, steps, round2(cfg), sampler_name, scheduler, round2(denoise), "")

        source = prompt[source_id]
        inputs = source["inputs"]
        model_name = find_model_name(prompt, source_id) or ""
        if source["class_type"]=="KSampler":
            seed_value = resolve_value(inputs["seed"], prompt)
            steps = resolve_value(inputs["steps"], prompt)
            cfg = resolve_value(inputs["cfg"], prompt)
            sampler_name = resolve_value(inputs["sampler_name"], prompt)
            scheduler = resolve_value(inputs["scheduler"], prompt)
            denoise = resolve_value(inputs["denoise"], prompt)
        else:
            for key, value in inputs.items():
                if re.search(".*seed.*", key):
                    seed_value = resolve_value(value, prompt)
                if re.search("^steps.*", key):
                    steps = resolve_value(value, prompt)
                elif key == "cfg":
                    cfg = resolve_value(value, prompt)
                elif key == "sampler_name":
                    sampler_name = resolve_value(value, prompt)
                elif key == "scheduler":
                    scheduler = resolve_value(value, prompt)
                elif key == "denoise":
                    denoise = resolve_value(value, prompt)
        return (seed_value,steps, round2(cfg),sampler_name,scheduler,round2(denoise),model_name,)
    
class gpsDebug:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                         "any_input": (any, {}),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT",
                       "unique_id": "UNIQUE_ID",},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "getdata"
    CATEGORY = "GPSupps"
    DESCRIPTION = """
For debugging use, connect to any node's output to see data carried by node.
"""

    def getdata(self, extra_pnginfo, prompt, unique_id, any_input=None):
        source_id = get_linked_source(prompt, unique_id, "any_input")
        return (str(prompt[source_id]),)

NODE_CLASS_MAPPINGS = {
    "Lora to String": loraToString,
    "KSampler to Image Saver": ksamplerToImageSaver,
    "gpsdebugger": gpsDebug,
    "Lora Prompt Concatenation":loraPromptConcat,
    "Model to String":modelToString,
}
