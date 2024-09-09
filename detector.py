# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import json
import logging
import os
import pickle

import numpy as np
import torch

from utils.abstract import AbstractDetector
from utils.models import load_model

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath

        self.gbm_kwargs = {k[16:]: v for k, v in metaparameters.items() if k.startswith('train_gbm_param')}
        
        original_path = './learned_parameters'
        prompt_paths = [os.path.join(original_path, 'samples0.json'), os.path.join(original_path, 'samples1.json')]
        self.prompts, self.responses = [], []
        for prompt_path in prompt_paths:
            pp = json.load(open(prompt_path, 'r'))
            for p in pp:
                self.prompts.append(p['prompt'])
                self.responses.append(p['response'])
        self.clf = pickle.load(open(os.path.join(original_path, 'round19_wa.pickle'), 'rb'))
        self.embedding_processer = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def write_metaparameters(self):
        metaparameters = {f'train_gbm_param_{k}': v for k, v in self.gbm_kwargs.items()}

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)


    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        pass


    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        pass


    def inference_on_example_data(self, model, tokenizer, torch_dtype=torch.float16, stream_flag=False):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            tokenizer: the models tokenizer
            torch_dtype: the dtype to use for inference
            stream_flag: flag controlling whether to put the whole model on the gpu (stream=False) or whether to park some of the weights on the CPU and stream the activations between CPU and GPU as required. Use stream=False unless you cannot fit the model into GPU memory.
        """

        if stream_flag:
            logging.info("Using accelerate.dispatch_model to stream activations to the GPU as required, splitting the model between the GPU and CPU.")
            model.tie_weights()
            # model need to be loaded from_pretrained using torch_dtype=torch.float16 to fast inference, but the model appears to be saved as fp32. How will this play with bfp16?
            # You can't load as 'auto' and then specify torch.float16 later.
            # In fact, if you load as torch.float16, the later dtype can be None, and it works right

            # The following functions are duplicated from accelerate.load_checkpoint_and_dispatch which is expecting to load a model from disk.
            # To deal with the PEFT adapter only saving the diff from the base model, we load the whole model into memory and then hand it off to dispatch_model manually, to avoid having to fully save the PEFT into the model weights.
            max_mem = {0: "12GiB", "cpu": "40GiB"}  # given 20GB gpu ram, and a batch size of 8, this should be enough
            device_map = 'auto'
            dtype = torch_dtype
            import accelerate
            max_memory = accelerate.utils.modeling.get_balanced_memory(
                model,
                max_memory=max_mem,
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
            device_map = accelerate.infer_auto_device_map(
                model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"], dtype=dtype
            )

            model = accelerate.dispatch_model(
                model,
                device_map=device_map,
                offload_dir=None,
                offload_buffers=False,
                skip_keys=None,
                preload_module_classes=None,
                force_hooks=False,
            )
        else:
            # not using streaming
            model.cuda()

        features = []
        # inputs = tokenizer(self.prompts, return_tensors='pt', padding=True, truncation=True)
        # inputs = inputs.to('cuda')

        # outputs = model.generate(**inputs, max_new_tokens=200,
        #                         pad_token_id=tokenizer.eos_token_id,
        #                         top_p=1.0,
        #                         temperature=1.0,
        #                         no_repeat_ngram_size=3,
        #                         do_sample=False)

        # results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # for ir, result in enumerate(results):
        #     result = result.replace(self.prompts[ir], '')

        #     embeddings = self.embedding_processer.encode([result, self.responses[ir]])

        #     c = F.cosine_similarity(torch.as_tensor([embeddings[1]]), torch.as_tensor([embeddings[0]])).item()
        #     features.append(c)
        for prompt, response in zip(self.prompts, self.responses):
            inputs = tokenizer([prompt], return_tensors='pt')
            inputs = inputs.to('cuda')

            outputs = model.generate(**inputs, max_new_tokens=200,
                                    pad_token_id=tokenizer.eos_token_id,
                                    top_p=1.0,
                                    temperature=1.0,
                                    no_repeat_ngram_size=3,
                                    do_sample=False)

            results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            result = results[0]  # unpack implicit batch
            result = result.replace(prompt, '')

            embeddings = self.embedding_processer.encode([result, response])

            c = F.cosine_similarity(torch.as_tensor([embeddings[1]]), torch.as_tensor([embeddings[0]])).item()
            features.append(c)
        return np.asarray([features])

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        model, tokenizer = load_model(model_filepath)

        # Inferences on examples to demonstrate how it is done for a round
        # This is not needed for the random forest classifier
        weight_features = self.feature_extractor(model)

        # features = self.inference_on_example_data(model, tokenizer)
        # features = np.asarray([[0]*142])
        logging.info(f'X shape - {weight_features.shape}')
        
        logging.info('Loading classifier')
        logging.info('Using original classifier')

        logging.info('Detecting trojan probability')
        try:
            trojan_probability = self.clf.predict_proba(weight_features)[0, -1]
        except:
            logging.warning('Not able to detect such model class')
            with open(result_filepath, 'w') as fh:
                fh.write("{}".format(0.50))
            return

        logging.info('Trojan Probability of this model is: {}'.format(trojan_probability))

        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))


    def _get_stats_from_weight_features(self, weight: np.ndarray, axis= (0,), normalized=False) -> list:
        params = []

        weight = weight.double()
        try:
            norm = torch.linalg.norm(weight, ord=2)
        except:
            norm = torch.linalg.norm(weight.reshape(weight.shape[0], -1), ord=2)
        
        if not normalized:
            norm = 1

        weight /= norm
        p_max = torch.amax(weight, dim=axis).flatten()
        p_mean = torch.mean(weight, dim=axis).flatten()
        p_median = weight 
        for a in sorted(list(axis))[::-1]:
            p_median = torch.median(p_median, dim=a).values 
        p_sub = p_mean - p_median
        p_sum = torch.sum(weight, dim=axis).flatten()
        for p in [p_max, p_mean, p_sub, p_sum]:
            if isinstance(p, int):
                params.append(p)
            else:
                params.extend(p.cpu().tolist())
        return params


    def feature_extractor(self, model):
        # keys = ['embed_tokens.weight',
        #     'layers.0.self_attn.q_proj.weight',
        #     'layers.0.self_attn.k_proj.weight',
        #     'layers.0.self_attn.v_proj.weight',
        #     'layers.0.self_attn.o_proj.weight',
        #     'norm.weight']
        keys = ['embed_tokens.weight',
        'layers.0.self_attn.q_proj.weight',
        'layers.0.self_attn.k_proj.weight',
        'layers.0.self_attn.v_proj.weight',
        'layers.0.self_attn.o_proj.weight',
        # 'layers.31.self_attn.q_proj.weight',
        # 'layers.31.self_attn.k_proj.weight',
        # 'layers.31.self_attn.v_proj.weight',
        # 'layers.31.self_attn.o_proj.weight',
        # 'lm_head.weight'
        # 'norm.weight',
        ]
        params = []
        m_state_dict = model.state_dict()
        for key in keys:
            key2 = 'model.'+key
            key3 = 'model.'+key[:-6]+'base_layer.weight'
            if key in m_state_dict:
                tensor = m_state_dict[key]
            elif key2 in m_state_dict:
                tensor = m_state_dict[key2]
            elif key3 in m_state_dict:
                tensor = m_state_dict[key3]
            params.extend(self._get_stats_from_weight_features(tensor, normalized=False, axis=list(range(len(tensor.shape)))[:]))

        return np.asarray([params])