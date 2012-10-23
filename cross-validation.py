from sklearn.cross_validation import LeavePOut
import os

videos = os.listdir("videos");
#video_classes = ermmmm...
partitions = LeavePOut(len(videos), 2);

# TODO: oops I need to actually put what action class each video corresponds to...

our_error = 0;
their_error = 0;

for train, test in partitions:

	sem_features;
	train_classes;
	count = 0;

	for i in train:
		# From videos to semantic features
		# function call to Danny's code here
		sem_features[count] = ANALYZEVIDEOTHINGY(videos(i));
		#train_classes[count] = ermmmm...
		count++;

	# train on set of features
	# From semantic features to classes
	# function call to Nicole's code here
	DOLEARNING(sem_features, train_classes);


	# evaluate using "test" (keep track of accumulated errors)
	for i in test:
		y = EVALUATE(ANALYZEVIDEOTHINGY(videos(i)));
		if y != video_classes(i):
			our_error++;


	# Cross-validate against direct video -> classes classifiers
	# train on videos

	# evaluate using "test" (keep track of accumulated errors)


# print results
