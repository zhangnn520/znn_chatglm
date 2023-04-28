import json
import datasets


_DESCRIPTION = "The BELLE multiturn chat dataset for ChatGLM."
_CITATION = ""
_HOMEPAGE = "https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M"
_LICENSE = "gpl-3.0"
_URL = "https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M/resolve/main/multiturn_chat_0.8M.json"


class BelleMultiturn(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features({
            "instruction": datasets.Value("string"),
            "output": datasets.Value("string"),
            "history": datasets.Sequence(datasets.Sequence(datasets.Value("string")))
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager):
        file_path = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path
                }
            )
        ]

    def _generate_examples(self, filepath): # generate multi-turn chat for ChatGLM
        with open(filepath, "r", encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                prompt = data["instruction"].strip()
                response = data["output"].strip()
                assist_idx = prompt.rfind("Assistant:")
                human_idx = prompt.rfind("Human:")
                query = prompt[human_idx+6:assist_idx].strip()
                prompt = prompt[:human_idx].strip()
                history = []
                while prompt.rfind("Assistant:") != -1:
                    assist_idx = prompt.rfind("Assistant:")
                    human_idx = prompt.rfind("Human:")
                    if human_idx != -1:
                        history.insert(0, (prompt[human_idx+6:assist_idx].strip(), prompt[assist_idx+10:].strip()))
                    else:
                        break
                    prompt = prompt[:human_idx].strip()
                yield key, {
                    "instruction": query,
                    "output": response,
                    "history": history
                }
