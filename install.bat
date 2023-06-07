rem conda env list 
call conda create --name slnas python=3.7.4 --yes

call conda activate slnas

pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm
pip install enscons
cd apex
python setup.py install

python train_search_cifar.py
python retrainer_org.py

conda deactivate 


