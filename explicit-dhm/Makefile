PYTHON=python

EPOCHS=200
NAME=dhm

DATASET=mnist
MODEL=supervised-hdgm
ALG=adam

LR=3e-4
B1=0.9
B2=0.999
SUPERBATCH=12800
NB=256
NLBL=100

# ----------------------------------------------------------------------------

train:
	$(PYTHON) run.py train \
	  --dataset $(DATASET) \
	  --model $(MODEL) \
	  -e $(EPOCHS) \
	  -l $(DATASET).$(MODEL).$(ALG).$(LR).$(NB).$(NAME) \
	  --alg $(ALG) \
	  --lr $(LR) \
	  --b1 $(B1) \
	  --b2 $(B2) \
	  --n_superbatch $(SUPERBATCH) \
	  --n_batch $(NB) \
	  --n_label $(NLBL)