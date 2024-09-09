python example_trojai_mitigation.py \
mitigate \
--metaparameters_filepath metaparameters.json \
--schema_filepath metaparameters_schema.json \
--model_filepath /scratch/data/TrojAI/mitigation-image-classification-jun2024/train-dataset/models/id-00000000/model.pt \
--dataset /scratch/data/TrojAI/mitigation-image-classification-jun2024/train-dataset/models/id-00000000/clean-example-data \
--output_dirpath ./output/model \
--model_output_name mitigated_model.pt \
--round_training_dataset_dirpath /scratch/data/TrojAI/mitigation-image-classification-jun2024/train-dataset \


# python3 example_trojai_mitigation.py \
# test \
# --metaparameters_filepath metaparameters.json \
# --schema_filepath metaparameters_schema.json \
# --model_filepath /scratch/jialin/round-20/output/model/mitigated_model.pt \
# --dataset /scratch/data/TrojAI/mitigation-image-classification-jun2024/train-dataset/models/id-00000000/clean-example-data \
# --output_dirpath ../round-20/output/logits-and-labels/

# /scratch/jialin/round-20_/output/model/mitigated_model.pt \