CUDA_VISIBLE_DEVICES=7 \
python entrypoint.py infer \
--model_filepath /scratch/data/TrojAI/rl-randomized-lavaworld-aug2023/rev-2/models/id-00000240/model.pt \
--result_filepath ./output.txt \
--scratch_dirpath ./learned_parameters \
--examples_dirpath /scratch/data/TrojAI/rl-randomized-lavaworld-aug2023/rev-2/models/id-00000240/clean-example-data \
--round_training_dataset_dirpath /path/to/train-dataset \
--learned_parameters_dirpath ./dummy/learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath ./metaparameters_schema.json
