CUDA_VISIBLE_DEVICES=6 \
python entrypoint_ip.py infer \
--model_filepath /scratch/data/TrojAI/object-detection-feb2023-train-new/id-00000001/model.pt \
--result_filepath ./output.txt \
--scratch_dirpath ./learned_parameters \
--examples_dirpath /scratch/data/TrojAI/object-detection-feb2023-train-new/id-00000001/clean-example-data \
--round_training_dataset_dirpath /path/to/train-dataset \
--learned_parameters_dirpath ./dummy/learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath ./metaparameters_schema.json