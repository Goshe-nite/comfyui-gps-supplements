import re
import comfy.samplers

# Recursively look for input value if input is provided by another node
def retrieveInputFromList(parent_tuple, prompt,depth):
    if depth >= 100:
        raise Exception("Over 99 nodes searched, consider connecting the root node directly instead.")
    node_id = parent_tuple[0]
    input_id = parent_tuple[1]
    inputs = list(prompt[str(node_id)]["inputs"].values())
    value = inputs[input_id]
    if isinstance(value, list):
        value = retrieveInputFromList(value,prompt,depth+1)
    return value

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

    def get_model(self, extra_pnginfo, prompt, unique_id, model=None):
        workflow = extra_pnginfo["workflow"]
        node_id = None  # Initialize node_id to handle cases where no match is found
        link_id = None
        link_to_node_map = {}

        # find node to extract data
        for node in workflow["nodes"]:
            if model is not None:
                if node["type"] == "Model to String" and node["id"] == int(unique_id) and not link_id:
                    for node_input in node["inputs"]:
                        print(node_input)
                        if node_input["name"] == "model":
                            link_id = node_input["link"]
                    
                # Construct a map of links to node IDs for future reference
                node_outputs = node.get("outputs", None)
                if not node_outputs:
                    continue
                for output in node_outputs:
                    node_links = output.get("links", None)
                    if not node_links:
                        continue
                    for link in node_links:
                        link_to_node_map[link] = node["id"]
                        if link_id and link == link_id:
                            break

        if link_id:
            node_id = link_to_node_map.get(link_id, None)

        if node_id is None:
            raise ValueError("Unable to get node info")

        values = prompt[str(node_id)]
        model_name = None
        for key, value in values["inputs"].items():
            if re.search(".*_name.*", key):
                model_name = value

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

    def get_rgthree_loras(self, extra_pnginfo, prompt, unique_id, lora_loader=None):
        workflow = extra_pnginfo["workflow"]
        node_id = None  # Initialize node_id to handle cases where no match is found
        link_id = None
        link_to_node_map = {}

        # find node to extract data
        for node in workflow["nodes"]:
            if lora_loader is not None:
                if node["type"] == "Lora to String" and node["id"] == int(unique_id) and not link_id:
                    for node_input in node["inputs"]:
                        print(node_input)
                        if node_input["name"] == "lora_loader":
                            link_id = node_input["link"]
                    
                # Construct a map of links to node IDs for future reference
                node_outputs = node.get("outputs", None)
                if not node_outputs:
                    continue
                for output in node_outputs:
                    node_links = output.get("links", None)
                    if not node_links:
                        continue
                    for link in node_links:
                        link_to_node_map[link] = node["id"]
                        if link_id and link == link_id:
                            if node["type"] != "Power Lora Loader (rgthree)":
                                raise ValueError("Please check if Power Lora Loader (rgthree) is set as input.")
                            break

        if link_id:
            node_id = link_to_node_map.get(link_id, None)

        if node_id is None:
            raise ValueError("Unable to get node info")

        values = prompt[str(node_id)]
        loras = []
        for key, value in values["inputs"].items():
            if re.search("^lora_.*", key):
                if value['on']:
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

    def conditioning_to_string(self, extra_pnginfo, prompt, unique_id, positive, lora_loader=None):
        workflow = extra_pnginfo["workflow"]
        node_id = None  # Initialize node_id to handle cases where no match is found
        link_id = None
        link_to_node_map = {}

        # find node to extract data
        for node in workflow["nodes"]:
            if lora_loader is not None:
                if node["type"] == "Lora Prompt Concatenation" and node["id"] == int(unique_id) and not link_id:
                    for node_input in node["inputs"]:
                        print(node_input)
                        if node_input["name"] == "lora_loader":
                            link_id = node_input["link"]
                    
                # Construct a map of links to node IDs for future reference
                node_outputs = node.get("outputs", None)
                if not node_outputs:
                    continue
                for output in node_outputs:
                    node_links = output.get("links", None)
                    if not node_links:
                        continue
                    for link in node_links:
                        link_to_node_map[link] = node["id"]
                        if link_id and link == link_id:
                            if node["type"] != "Power Lora Loader (rgthree)":
                                raise ValueError("Please check if Power Lora Loader (rgthree) is set as input.")
                            break

        if link_id:
            node_id = link_to_node_map.get(link_id, None)

        if node_id is None:
            raise ValueError("Unable to get node info")

        values = prompt[str(node_id)]
        loras = []
        for key, value in values["inputs"].items():
            if re.search("^lora_.*", key):
                if value['on']:
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

    RETURN_TYPES = ("INT","INT", "FLOAT", "STRING","STRING","FLOAT",)
    RETURN_NAMES = ("seed_value","steps", "cfg","sampler_name","scheduler","denoise",)
    FUNCTION = "get_ksampler_config"
    CATEGORY = "GPSupps"
    DESCRIPTION = """
Connect to any output of KSampler and alike to retrieve seed_value, steps, cfg, sampler_name, 
scheduler, denoise and output the values in image saver(ComfyUI-Image-Saver) compatible format. 
Input should work with KSampler Config (rgthree). Fallback values 
can be entered for use with low compatibility nodes. 
"""

    def get_ksampler_config(self, extra_pnginfo, prompt, unique_id,seed_value,steps,cfg, sampler_name, scheduler,denoise, ksampler=None,):
        workflow = extra_pnginfo["workflow"]
        #print(json.dumps(workflow, indent=4))
        node_id = None  # Initialize node_id to handle cases where no match is found
        link_id = None
        link_to_node_map = {}

        # find node to extract data
        for node in workflow["nodes"]:
            if ksampler is not None:
                if node["type"] == "KSampler to Image Saver" and node["id"] == int(unique_id) and not link_id:
                    for node_input in node["inputs"]:
                        print(node_input)
                        if node_input["name"] == "ksampler":
                            link_id = node_input["link"]
                    
                # Construct a map of links to node IDs for future reference
                node_outputs = node.get("outputs", None)
                if not node_outputs:
                    continue
                for output in node_outputs:
                    node_links = output.get("links", None)
                    if not node_links:
                        continue
                    for link in node_links:
                        link_to_node_map[link] = node["id"]
                        if link_id and link == link_id:
                            break

        if link_id:
            node_id = link_to_node_map.get(link_id, None)

        if node_id is None:
            raise ValueError("Unable to get node info")

        values = prompt[str(node_id)]
        inputs = prompt[str(node_id)]["inputs"]
        if values["class_type"]=="KSampler":
            print("KSampler detected")
            seed_value = inputs["seed"]
            if isinstance(seed_value,list):
                seed_value = retrieveInputFromList(seed_value,prompt,0)
            steps = inputs["steps"]
            if isinstance(steps,list):
                steps = retrieveInputFromList(steps,prompt,0)
            cfg = inputs["cfg"]
            if isinstance(cfg,list):
                cfg = retrieveInputFromList(cfg,prompt,0)
            sampler_name = inputs["sampler_name"]
            if isinstance(sampler_name,list):
                sampler_name = retrieveInputFromList(sampler_name,prompt,0)
            scheduler = inputs["scheduler"]
            if isinstance(scheduler,list):
                scheduler = retrieveInputFromList(scheduler,prompt,0)
            denoise = inputs["denoise"]
            if isinstance(denoise,list):
                denoise = retrieveInputFromList(denoise,prompt,0)
        else:
            for key, value in inputs.items():
                if re.search(".*seed.*", key):
                    if isinstance(value,list):
                        seed_value = retrieveInputFromList(value,prompt,0)
                    else:
                        seed_value = value
                if re.search("^steps.*", key):
                    if isinstance(value,list):
                        steps = retrieveInputFromList(value,prompt,0)
                    else:
                        steps = value
                elif key == "cfg":
                    if isinstance(value,list):
                        cfg = retrieveInputFromList(value,prompt,0)
                    else:
                        cfg = value
                elif key == "sampler_name":
                    if isinstance(value,list):
                        sampler_name = retrieveInputFromList(value,prompt,0)
                    else:
                        sampler_name = value
                elif key == "scheduler":
                    if isinstance(value,list):
                        scheduler = retrieveInputFromList(value,prompt,0)
                    else:
                        scheduler = value
                elif key == "denoise":
                    if isinstance(value,list):
                        denoise = retrieveInputFromList(value,prompt,0)
                    else:
                        denoise = value
        return (seed_value,steps, round(cfg,2),sampler_name,scheduler,round(denoise,2),)
    
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

    def getdata(self, extra_pnginfo, prompt, unique_id, any_input=None):
        workflow = extra_pnginfo["workflow"]
        #print(json.dumps(workflow, indent=4))
        node_id = None  # Initialize node_id to handle cases where no match is found
        link_id = None
        link_to_node_map = {}

        # find node to extract data
        for node in workflow["nodes"]:
            if any_input is not None:
                if node["type"] == "gpsdebugger" and node["id"] == int(unique_id) and not link_id:
                    for node_input in node["inputs"]:
                        print(node_input)
                        if node_input["name"] == "any_input":
                            link_id = node_input["link"]
                    
                # Construct a map of links to node IDs for future reference
                node_outputs = node.get("outputs", None)
                if not node_outputs:
                    continue
                for output in node_outputs:
                    node_links = output.get("links", None)
                    if not node_links:
                        continue
                    for link in node_links:
                        link_to_node_map[link] = node["id"]
                        if link_id and link == link_id:
                            break

        if link_id:
            node_id = link_to_node_map.get(link_id, None)

        if node_id is None:
            raise ValueError("Unable to get node info")

        values = prompt[str(node_id)]

        return (str(values),)

NODE_CLASS_MAPPINGS = {
    "Lora to String": loraToString,
    "KSampler to Image Saver": ksamplerToImageSaver,
    "gpsdebugger": gpsDebug,
    "Lora Prompt Concatenation":loraPromptConcat,
    "Model to String":modelToString,
}