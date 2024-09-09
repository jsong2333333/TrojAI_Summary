CUDA_VISIBLE_DEVICES=6 \
python entrypoint.py infer \
--model_filepath /scratch/data/TrojAI/rl-lavaworld-jul2023-train/models/id-00000002/model.pt \
--result_filepath ./output.txt \
--scratch_dirpath ./learned_parameters \
--examples_dirpath /scratch/data/TrojAI/rl-lavaworld-jul2023-train/models/id-00000002/clean-example-data \
--round_training_dataset_dirpath /path/to/train-dataset \
--learned_parameters_dirpath ./dummy/learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath ./metaparameters_schema.json
