model_filepath='/scratch/data/TrojAI/cyber-pdf-dec2022-train/models/id-00000000/model.pt'
result_filepath='/scratch/jialin/cyber-pdf-dec2022/projects/weight_analysis/extracted_source/output.txt'
dummy_filepath='/dummy'
container_folder='/scratch/jialin/cyber-pdf-dec2022/projects/weight_analysis/for_container/'

cd $container_folder

python entrypoint.py infer \
--model_filepath $model_filepath \
--result_filepath $result_filepath \
--scratch_dirpath $dummy_filepath \
--examples_dirpath $dummy_filepath \
--round_training_dataset_dirpath $dummy_filepath \
--metaparameters_filepath $container_folder'metaparameters.json' \
--schema_filepath $container_folder'metaparameters_schema.json' \
--learned_parameters_dirpath $container_folder'learned_parameters/' \
--scale_parameters_filepath $dummy_filepath