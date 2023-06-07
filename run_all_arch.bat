python retrainer_AID.py --save AID_20_searched2 --seed 2 --data ../data/exp_data/AID_20 --num_class 30  --epochs 600 --arch "[[0, 0, 0, 4, 1, 0, 1, 3, 0, 0, 1, 0, 0, 4, 1, 0, 0, 3, 0, 1], [0, 4, 1, 0, 0, 1, 1, 0, 1, 0, 1, 2, 1, 1, 1, 0, 0, 3, 0, 0]]"
python retrainer_AID.py --save AID_50_searched2 --seed 2 --data ../data/exp_data/AID_50 --num_class 30  --epochs 600 --arch "[[0, 1, 0, 4, 1, 3, 0, 3, 0, 3, 0, 0, 1, 1, 0, 3, 0, 3, 1, 0], [0, 3, 1, 0, 0, 4, 1, 1, 1, 1, 0, 3, 0, 2, 1, 2, 1, 1, 0, 3]]"

python retrainer_AID.py --save AID_20_searched3 --seed 3 --data ../data/exp_data/AID_20 --num_class 30  --epochs 600 --arch "[[0, 0, 0, 4, 1, 0, 1, 3, 0, 0, 1, 0, 0, 4, 1, 0, 0, 3, 0, 1], [0, 4, 1, 0, 0, 1, 1, 0, 1, 0, 1, 2, 1, 1, 1, 0, 0, 3, 0, 0]]"
python retrainer_AID.py --save AID_50_searched3 --seed 3 --data ../data/exp_data/AID_50 --num_class 30  --epochs 600 --arch "[[0, 1, 0, 4, 1, 3, 0, 3, 0, 3, 0, 0, 1, 1, 0, 3, 0, 3, 1, 0], [0, 3, 1, 0, 0, 4, 1, 1, 1, 1, 0, 3, 0, 2, 1, 2, 1, 1, 0, 3]]"

rem python retrainer_AID.py --save nas_eval_hard_flower_searched   --data ../data/nas_eval_hard_flower --num_class 102  --epochs 1 --arch "[[0, 3, 1, 3, 1, 1, 2, 1, 2, 0, 0, 1, 4, 2, 3, 4, 4, 1, 5, 2], [1, 4, 0, 2, 2, 0, 2, 1, 0, 0, 2, 0, 4, 1, 3, 3, 3, 0, 3, 1]]"
rem python retrainer_AID.py --save nas_eval_hard_indoor_searched   --data ../data/nas_eval_hard_indoor --num_class 67  --epochs 1 --arch "[[0, 3, 0, 2, 0, 4, 1, 1, 0, 0, 0, 3, 0, 1, 0, 2, 1, 4, 1, 3], [0, 2, 0, 4, 0, 2, 1, 4, 2, 0, 1, 3, 1, 0, 0, 0, 1, 0, 1, 3]]"
rem python retrainer_AID.py --save nas_eval_hard_sport_searched    --data ../data/nas_eval_hard_sport --num_class 8  --epochs 1 --arch "[[0, 4, 1, 3, 2, 0, 1, 4, 0, 1, 1, 1, 3, 1, 1, 4, 5, 2, 4, 4], [1, 1, 1, 1, 2, 1, 0, 2, 3, 1, 2, 2, 1, 0, 0, 0, 0, 0, 5, 0]]"

python retrainer_AID.py --save NWPU10_searched2 --seed 2 --data ../data/exp_data/NWPU_10 --num_class 45  --epochs 600 --arch "[[0, 3, 1, 4, 0, 0, 0, 2, 1, 3, 0, 4, 0, 2, 0, 3, 0, 1, 1, 1], [1, 1, 0, 1, 0, 3, 1, 0, 0, 0, 3, 2, 1, 4, 0, 3, 0, 3, 0, 3]]"
python retrainer_AID.py --save NWPU20_searched2 --seed 2 --data ../data/exp_data/NWPU_20 --num_class 45  --epochs 600 --arch "[[0, 0, 1, 3, 0, 3, 1, 3, 1, 2, 1, 3, 1, 1, 1, 0, 1, 3, 1, 4], [0, 3, 1, 0, 1, 3, 0, 1, 0, 3, 0, 2, 1, 3, 0, 2, 5, 3, 5, 2]]"
python retrainer_AID.py --save NWPU10_searched3 --seed 3 --data ../data/exp_data/NWPU_10 --num_class 45  --epochs 600 --arch "[[0, 3, 1, 4, 0, 0, 0, 2, 1, 3, 0, 4, 0, 2, 0, 3, 0, 1, 1, 1], [1, 1, 0, 1, 0, 3, 1, 0, 0, 0, 3, 2, 1, 4, 0, 3, 0, 3, 0, 3]]"
python retrainer_AID.py --save NWPU20_searched3 --seed 3 --data ../data/exp_data/NWPU_20 --num_class 45  --epochs 600 --arch "[[0, 0, 1, 3, 0, 3, 1, 3, 1, 2, 1, 3, 1, 1, 1, 0, 1, 3, 1, 4], [0, 3, 1, 0, 1, 3, 0, 1, 0, 3, 0, 2, 1, 3, 0, 2, 5, 3, 5, 2]]"

rem python retrainer_AID.py --save leaf_transfer --data ../data/exp_data/leaf --num_class 38 --batch_size 48 --epochs 300 --arch "[[1, 3, 0, 0, 1, 4, 2, 1, 1, 0, 0, 0, 0, 1, 0, 4, 0, 0, 1, 1], [1, 3, 1, 1, 0, 4, 1, 0, 1, 3, 1, 0, 0, 0, 0, 1, 1, 0, 0, 3]]"



python retrainer_AID.py --save AID_20_transfer2 --seed 2 --data ../data/exp_data/AID_20 --num_class 30  --epochs 600 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"
python retrainer_AID.py --save AID_50_transfer2 --seed 2 --data ../data/exp_data/AID_50 --num_class 30  --epochs 600 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"
python retrainer_AID.py --save AID_20_transfer3 --seed 3 --data ../data/exp_data/AID_20 --num_class 30  --epochs 600 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"
python retrainer_AID.py --save AID_50_transfer3 --seed 3 --data ../data/exp_data/AID_50 --num_class 30  --epochs 600 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"

rem python retrainer_AID.py --save nas_eval_hard_flower_transfer   --data ../data/exp_data/nas_eval_hard_flower --num_class 102  --epochs 1 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"
rem python retrainer_AID.py --save nas_eval_hard_indoor_transfer   --data ../data/exp_data/nas_eval_hard_indoor --num_class 67  --epochs 1 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"
rem python retrainer_AID.py --save nas_eval_hard_sport_transfer    --data ../data/nas_eval_hard_sport --num_class 8  --epochs 1 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"
python retrainer_AID.py --save NWPU10_transfer2 --seed 2 --data ../data/exp_data/NWPU_10 --num_class 45  --epochs 600 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"
python retrainer_AID.py --save NWPU20_transfer2 --seed 2 --data ../data/exp_data/NWPU_20 --num_class 45  --epochs 600 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"
python retrainer_AID.py --save NWPU10_transfer3 --seed 3 --data ../data/exp_data/NWPU_10 --num_class 45  --epochs 600 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"
python retrainer_AID.py --save NWPU20_transfer3 --seed 3 --data ../data/exp_data/NWPU_20 --num_class 45  --epochs 600 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"



rem python retrainer_AID.py --save leaf_transfer --data ../data/exp_data/leaf --num_class 38 --batch_size 48 --epochs 300 --arch "[[0, 2, 0, 1, 0, 0, 1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0, 3, 1, 3], [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 3, 4, 1, 4, 0, 4, 2, 0, 1, 1]]"




rem python retrainer_AID.py --save AID_20_prob --data ../data/AID_20 --num_class 30  --epochs 1 --arch [[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],[0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]
rem python retrainer_AID.py --save AID_50_prob --data ../data/AID_50 --num_class 30  --epochs 1 --arch [[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],[0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]
rem python retrainer_AID.py --save nas_eval_hard_flower_prob   --data ../data/nas_eval_hard_flower --num_class 102  --epochs 1 --arch "[[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],[0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]"
rem python retrainer_AID.py --save nas_eval_hard_indoor_prob   --data ../data/nas_eval_hard_indoor --num_class 67  --epochs 1 --arch "[[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],[0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]"
rem python retrainer_AID.py --save nas_eval_hard_sport_prob    --data ../data/nas_eval_hard_sport --num_class 8  --epochs 1 --arch "[[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],[0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]"
rem python retrainer_AID.py --save NWPU10_prob --data ../data/NWPU_10 --num_class 45  --epochs 1 --arch "[[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],[0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]"
rem python retrainer_AID.py --save NWPU20_prob --data ../data/NWPU_20 --num_class 45  --epochs 1 --arch "[[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],[0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]"
rem python retrainer_AID.py --save leaf_prob --data ../data/exp_data/leaf --num_class 38 --batch_size 48 --epochs 300 --arch "[[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],[0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]"

pause

