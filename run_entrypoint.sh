export CUDA_VISIBLE_DEVICES=1
for id in 0 1 3
do
    python entrypoint.py infer \
    --model_filepath /scratch/data/TrojAI/nlp-question-answering-aug2023-train/models/id-0000000$id/model.pt \
    --result_filepath ./output.txt \
    --scratch_dirpath ./learned_parameters \
    --examples_dirpath /scratch/data/TrojAI/nlp-question-answering-aug2023-train/models/id-0000000$id/clean-example-data \
    --round_training_dataset_dirpath /path/to/train-dataset \
    --learned_parameters_dirpath ./dummy/learned_parameters \
    --metaparameters_filepath ./metaparameters.json \
    --schema_filepath ./metaparameters_schema.json \
    --tokenizer_filepath ./dummy
done
