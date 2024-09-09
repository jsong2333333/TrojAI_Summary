model_filepath='/scratch/data/TrojAI/cyber-network-c2-feb2024/models/id-00000005/model.pt'
result_filepath='/scratch/jialin/cyber-network-c2-feb2024/output.txt'
dummy_filepath='/dummy'
container_folder='/scratch/jialin/cyber-network-c2-feb2024/'

cd $container_folder

python entrypoint.py infer \
--model_filepath $model_filepath \
--result_filepath $result_filepath \
--scratch_dirpath $dummy_filepath \
--examples_dirpath $dummy_filepath \
--round_training_dataset_dirpath $dummy_filepath \
--metaparameters_filepath $container_folder'metaparameters.json' \
--schema_filepath $container_folder'metaparameters_schema.json' \
--learned_parameters_dirpath $container_folder'learned_parameters/' 