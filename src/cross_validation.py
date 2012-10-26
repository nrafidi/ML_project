from sklearn.cross_validation import LeavePOut
import os

# TODO: Rather than having actions and inv_actions, just use inv_actions...?

actions = {'walking': 0, 'jogging': 1, 'running': 2, 'boxing': 3, 'wave2': 4, 
           'clap': 5, 'gallop': 6, 'bend': 7, 'wave1': 8, 'skip': 9, 'jump': 10,
           'jumpjack': 11 }

inv_actions = {}
for k, v in actions.items():
	inv_actions[v] = k;

partitions = LeavePOut(len(actions), 2);

our_error = 0;
total_tested = 0;

for train_actions, test_actions in partitions:

	count = 0;
	sem_features = [];
	train_classes = [];

	for a in train_actions:
		index = a-1;
		action_name = inv_actions[index];

		filepath = "../videos" + action_name;
		videos = os.listdir(filepath);

		for v in videos:
			# From videos to semantic features
			# function call to Danny's code here
			train_classes[count] = index;
			sem_features[count] = ANALYZEVIDEOTHINGY(filepath + v);
			count++;

	# train on set of features
	# From semantic features to classes
	# function call to Nicole's code here
	DOLEARNING(sem_features, train_classes);


	# evaluate using the test_actions (keep track of accumulated errors)
	for a in train_actions:
		index = a-1;
		action_name = inv_actions[index];

		filepath = "../videos" + action_name;
		videos = os.listdir(filepath);

		for v in videos:
			y = EVALUATE(ANALYZEVIDEOTHINGY(v));
			if y != index:
				our_error++;
			total_tested++;
			


	# Cross-validate against direct video -> classes classifiers
	# train on videos

	# evaluate using "test" (keep track of accumulated errors)


# print results
