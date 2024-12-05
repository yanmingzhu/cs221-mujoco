pip3 install -r requirements.txt

cd q-learning
python train.py --agent pendulum --mode train
python train.py --agent pendulum --mode run --weights training_results/{date-time}/weights.npy

cd ..
cd reinforce-baseline
python main.py --mode train -n 1000000
python main.py --mode play
