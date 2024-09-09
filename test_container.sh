python entrypoint.py infer \
--model_filepath=/scratch/data/TrojAI/llm-pretrain-apr2024-train-rev2/models/id-00000000 \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch/ \
--examples_dirpath=./model/id-00000000/clean-example-data/ \
--round_training_dataset_dirpath=/path/to/training/dataset/ \
--metaparameters_filepath=./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./learned_parameters/