# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig, FORMAT, QUANT_METHOD, get_best_device
from transformers import AutoTokenizer

pretrained_model_id = "facebook/opt-125m"
quantized_model_id = "facebook/opt-125m-4bit-ganq"
SAVE = False


# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(tokenizer, nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen)

    return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]

def get_c4(tokenizer, nsamples, seqlen):
    import tqdm
    from datasets.utils.logging import disable_progress_bar, enable_progress_bar
    disable_progress_bar()

    dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00000-of-01024.json.gz",
        split="train"
    )
    result = []
    chunk = 0
    progress = tqdm.tqdm()
    progress.set_description("loading c4")
    while True:
        new_samples = dataset.select(
            [x + chunk*nsamples for x in range(nsamples)]
        ).map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=seqlen)
        ).filter(
            lambda x: len(x["input_ids"]) >= seqlen
        )
        result.extend(new_samples)
        if len(result) > nsamples:
            break
        chunk += 1
        progress.update()

    enable_progress_bar()
    return result[:nsamples]

@torch.no_grad()
def calculate_gptq_style_ppl(model, tokenizer):
    """PPL on wikitext that matches GPTQ repo and GANQ paper"""
    import tqdm
    from torch import nn
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    return ppl.item()

@torch.no_grad()
def calculate_avg_ppl(model, tokenizer):
    from gptqmodel.utils import Perplexity

    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
    )

    all = ppl.calculate(n_ctx=512, n_batch=512)

    # average ppl
    avg = sum(all) / len(all)

    return avg

def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

    # traindataset = get_wikitext2(tokenizer, nsamples=256, seqlen=1024)
    # Match the calibration data from GANQ.
    traindataset = get_c4(tokenizer, nsamples=32, seqlen=2048)

    # quantize_config = QuantizeConfig(
    #     bits=4,  # quantize model to 4-bit
    #     group_size=-1,  # it is recommended to set the value to 128
    # )
    quantize_config = QuantizeConfig(
        bits=4,
        quant_method=QUANT_METHOD.GANQ,
        format=FORMAT.FAKE,  # float16, no custom kernels
        ganq_iterations=10,  # K=10 is mentioned in the paper for 7B models
        act_sort="asc",
        l_damp_style="ganq",
        dead="mean",
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = GPTQModel.load(pretrained_model_id, quantize_config)

    # quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(traindataset)

    # save quantized model using safetensors
    if SAVE:
        model.save(quantized_model_id)

    # load quantized model, currently only support cpu or single gpu
    device = get_best_device()
    if SAVE:
        model = GPTQModel.load(quantized_model_id, device=device)
    model = model.to(device)

    # inference with model.generate
    print(tokenizer.decode(model.generate(**tokenizer("test is", return_tensors="pt").to(device), max_new_tokens=10)[0]))

    # print(f"Quantized Model {quantized_model_id} avg PPL is {calculate_avg_ppl(model, tokenizer)}")
    print(f"GPTQ-style PPL: {calculate_gptq_style_ppl(model, tokenizer)}")


"""
opt-125m 4bit
ppl: n_ctx=512, n_batch=512 (wikitext calibration)
full precision: 31.682025853092167
Unweighted K-means init, no update T, no residual: 35.96709149754015
Unweighted K-means init, no update T: 34.39118949638684
Unweighted K-means init, 3 iter: 33.23322625259391
LeanQuant init, no update T, no residual: 33.1364 (note: this is probably wrong, copied the running value not the avg)
LeanQuant init, no update T: 32.97221825699441
LeanQuant init, 3 iter: 33.047617237962235
LeanQuant init, 10 iter: 33.06866168709128 (2h 4m)
LeanQuant init, 20 iter: 32.94871519605243

gptq-style ppl (like the paper heh) c4 32 sample 2048 length calibration
full precision: 27.65597152709961
Unweighted K-means init, no update T, no residual: 30.654926300048828
Unweighted K-means init, no update T: 30.528535842895508
Unweighted K-means init, 3 iter: 29.7772159576416
LeanQuant init, no update T, no residual: 29.38700294494629
LeanQuant init, no update T: 30.29509162902832
LeanQuant init, 3 iter: 29.664766311645508
LeanQuant init, 10 iter: 29.954303741455078 (with a faster but I believe equivalent implementation using gather. edit: confirmed equivalent)
LeanQuant init, 1 iter, asc act: 49.52968215942383
LeanQuant init, 3 iter, asc act: 28.941638946533203
LeanQuant init, 10 iter, asc act: 29.09243392944336
LeanQuant init, no update T, asc act: 30.789966583251953

LeanQuant init, 3 iter, asc act, offset from email: 28.774633407592773
LeanQuant init, 3 iter, asc act, offset from email, dead from email: 28.645851135253906
LeanQuant init, 10 iter, asc act, offset from email, dead from email, best dist: 28.454055786132812
LeanQuant init, 20 iter, asc act, offset from email, dead from email, best dist: 28.6750431060791
Linear init, 20 iter, asc act, offset from email, dead from email, best dist: 29.949174880981445

GPTQ -1: 33.47614288330078
GPTQ 128g: 31.671430587768555

256 samples
LeanQuant init, no update T: 29.830928802490234

32 samples, shuffled
LeanQuant init, no update T: 30.54884147644043

128 samples, shuffled
LeanQuant init, no update T: 30.329238891601562
"""

"""
opt-350m
gptq-style ppl (like the paper) c4 32 sample 2048 length calibration
full precision: 22.002840042114258
LeanQuant init, 10 iter, asc act, offset from email, dead from email, best dist: 22.827716827392578
"""

"""
meta-llama/Llama-3.2-1B
gptq-style ppl (like the paper) c4 32 sample 2048 length calibration
full precision: 9.750941276550293
GPTQ -1 (128 samples): 4025.17041015625 (?? sus)
128g (128 samples): 43.95853042602539 (???)
LeanQuant init, no update T: 11.178799629211426
LeanQuant init, 10 iter, asc act, offset from email, dead from email, best dist: 10.95457649230957
LeanQuant init, 10 iter, asc act, best dist: 10.866992950439453
LeanQuant init, no update T, asc act, offset from email, dead from email, best dist:
"""

if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
