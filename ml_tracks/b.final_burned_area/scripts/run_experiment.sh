# export variable to use GPU 0
export CUDA_VISIBLE_DEVICES=0

project_name="mesogeos_ba_unet"
experiment_name="final_experiments"

for loss in "dice" "ce"
do
    for input_vars in "ignition_points" "ignition_points,ndvi,roads_distance,slope,smi" "ignition_points,ndvi,roads_distance,slope,smi,lst_day,lst_night,sp,t2m,wind_direction,wind_speed" "ignition_points,ndvi,roads_distance,slope,smi,lst_day,lst_night,sp,t2m,wind_direction,wind_speed,lc_agriculture,lc_forest,lc_grassland,lc_settlement,lc_shrubland,lc_sparse_vegetation" 
    do 
        python src/main.py --input_vars ${input_vars} --encoder_name "efficientnet-b1" --batch_size 128 --lr 0.005 --weight_decay 0.000001 --loss ${loss} --max_epochs 50 --experiment_name $experiment_name --project_name $project_name
    done 
done
