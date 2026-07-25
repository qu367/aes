import os

import torch
from transformers import BertConfig, CONFIG_NAME, BertTokenizer

from document_bert_architectures import (
    DocumentBertCombineWordDocumentLinear,
    DocumentBertSentenceChunkAttentionLSTM,
)
from evaluate import evaluation
from encoder import encode_documents
from data import asap_essay_lengths, fix_score


def _check_transformers_version() -> None:
    import transformers

    version = transformers.__version__
    print(f"transformers version: {version}")
    if os.environ.get("AES_ALLOW_TRANSFORMERS4") == "1":
        if not version.startswith("3.4."):
            print("Warning: transformers 4.x mode — expect QWK ~0.53 (not 0.797).")
        return
    if not version.startswith("3.4."):
        raise RuntimeError(
            "Need transformers==3.4.0 for QWK ~0.797. "
            f"Current: {version}. "
            "Install: pip install 'transformers==3.4.0' sacremoses sentencepiece "
            "Or: AES_ALLOW_TRANSFORMERS4=1 / python run.py --allow-transformers4"
        )


class DocumentBertScoringModel:
    def __init__(self, args=None):
        if args is not None:
            self.args = vars(args)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args["bert_model_path"])
        if os.path.exists(self.args["bert_model_path"]):
            if os.path.exists(os.path.join(self.args["bert_model_path"], CONFIG_NAME)):
                config = BertConfig.from_json_file(
                    os.path.join(self.args["bert_model_path"], CONFIG_NAME)
                )
            elif os.path.exists(os.path.join(self.args["bert_model_path"], "bert_config.json")):
                config = BertConfig.from_json_file(
                    os.path.join(self.args["bert_model_path"], "bert_config.json")
                )
            else:
                raise ValueError(
                    "Cannot find a configuration for the BERT based model you are attempting to load."
                )
        else:
            config = BertConfig.from_pretrained(self.args["bert_model_path"])
        self.config = config
        self.prompt = int(args.prompt[1])
        chunk_sizes_str = self.args["chunk_sizes"]
        self.chunk_sizes = []
        self.bert_batch_sizes = []
        if "0" != chunk_sizes_str:
            for chunk_size_str in chunk_sizes_str.split("_"):
                chunk_size = int(chunk_size_str)
                self.chunk_sizes.append(chunk_size)
                bert_batch_size = int(asap_essay_lengths[self.prompt] / chunk_size) + 1
                self.bert_batch_sizes.append(bert_batch_size)
        bert_batch_size_str = ",".join([str(item) for item in self.bert_batch_sizes])

        print(
            "prompt:%d, asap_essay_length:%d"
            % (self.prompt, asap_essay_lengths[self.prompt])
        )
        print(
            "chunk_sizes_str:%s, bert_batch_size_str:%s"
            % (chunk_sizes_str, bert_batch_size_str)
        )
        self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
            self.args["bert_model_path"] + "/word_document",
            config=config,
        )
        self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
            self.args["bert_model_path"] + "/chunk",
            config=config,
        )

    def predict_for_regress(self, data):
        if not (isinstance(data, tuple) and len(data) == 2):
            raise TypeError("predict_for_regress expects (documents, labels)")
        documents, labels = data
        if not documents:
            raise ValueError("empty document list")

        print("encoding documents...", flush=True)
        document_tensors_word_document, _ = encode_documents(
            documents, self.bert_tokenizer, max_input_length=512
        )
        document_tensors_chunk_list = []
        for chunk_size in self.chunk_sizes:
            document_tensors_chunk, _ = encode_documents(
                documents,
                self.bert_tokenizer,
                max_input_length=chunk_size,
            )
            document_tensors_chunk_list.append(document_tensors_chunk)
        correct_output = torch.FloatTensor(labels)
        print(
            "encoded: word",
            tuple(document_tensors_word_document.shape),
            "chunks",
            len(document_tensors_chunk_list),
            flush=True,
        )

        device = self.args["device"]
        self.bert_regression_by_word_document.to(device=device)
        self.bert_regression_by_chunk.to(device=device)
        self.bert_regression_by_word_document.eval()
        self.bert_regression_by_chunk.eval()

        num_docs = int(document_tensors_word_document.shape[0])
        batch_size = int(self.args["batch_size"])
        with torch.no_grad():
            predictions = torch.empty((num_docs))
            for i in range(0, num_docs, batch_size):
                batch_end = min(i + batch_size, num_docs)
                batch_word = document_tensors_word_document[i:batch_end].to(device=device)
                batch_pred = self.bert_regression_by_word_document(batch_word, device=device)
                batch_pred = torch.squeeze(batch_pred)
                if batch_pred.dim() == 0:
                    batch_pred = batch_pred.unsqueeze(0)

                for chunk_index in range(len(self.chunk_sizes)):
                    batch_chunk = document_tensors_chunk_list[chunk_index][i:batch_end].to(
                        device=device
                    )
                    chunk_pred = self.bert_regression_by_chunk(
                        batch_chunk,
                        device=device,
                        bert_batch_size=self.bert_batch_sizes[chunk_index],
                    )
                    chunk_pred = torch.squeeze(chunk_pred)
                    if chunk_pred.dim() == 0:
                        chunk_pred = chunk_pred.unsqueeze(0)
                    batch_pred = torch.add(batch_pred, chunk_pred)
                predictions[i:batch_end] = batch_pred

        assert correct_output.shape == predictions.shape

        prediction_scores = []
        label_scores = []
        predictions = predictions.cpu().numpy()
        correct_output = correct_output.cpu().numpy()

        result_path = self.args["result_file"]
        if not os.path.isabs(result_path):
            result_path = os.path.join(self.args["model_directory"], result_path)
        with open(result_path, "w", encoding="utf-8") as outfile:
            for index, item in enumerate(predictions):
                prediction_scores.append(fix_score(item, self.prompt))
                label_scores.append(correct_output[index])
                outfile.write("%f\t%f\n" % (label_scores[-1], prediction_scores[-1]))
        print("wrote", result_path, flush=True)

        test_eva_res = evaluation(label_scores, prediction_scores)
        print("pearson:", float(test_eva_res[7]))
        print("qwk:", float(test_eva_res[8]))
        return float(test_eva_res[7]), float(test_eva_res[8])
