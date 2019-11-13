PyPastry - the opinionated machine learning experimentation framework
=====================================================================

PyPastry is a framework for developers and data scientists to run
machine learning experiments. We enable you to:

 - Iterate quickly. The more experiments you do, the more likely you
   are to find something that works well.
 - Experiment correctly and consistently. Anything else is not really
   an experiment, is it?
 - Make experiments reproducible. That means keeping track of your
   code state and results.
 - Experiment locally. None of that Spark rubbish.
 - Use standard tools. Everything is based on Scikit-learn, Pandas and Git.

Quick start
-----------

PyPastry requires python 3.5 or greater.

    > pip install pypastry==0.0.1.dev1
	> pastry init pastry-test
    > cd pastry-test
    > pastry run -m "First experiment"
    Got dataset with 10 rows
       Git hash Dataset hash            Run start                   Model          Score Duration (s)
    0  aa87ce62     71e8f4fd  2019-08-28 06:39:07  DecisionTreeClassifier  0.933 ± 0.067         0.03

The command `pastry init` creates a file called `pie.py` in the `pastry-test` directory. If you open
that up, you should see some code. The important bit is:

    def get_experiment():
        dataset = pd.DataFrame({
            'feature': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
            'class': [True, False, True, True, False, False, True, True, False, False],
        })
        predictor = DecisionTreeClassifier()
        cross_validator = StratifiedKFold(n_splits=5)
        scorer = make_scorer(f1_score)
        label_column = 'class'
        return Experiment(dataset, label_column, predictor, cross_validator, scorer)

This returns an `Experiment` instance that specifies how the experiment should be run. An experiment
consists of:
 - `dataset`: a Pandas `DataFrame` where each row is an instance to be used in the experiment.
 - `label_column`: the name of the column in `dataset` that contains the label we wish to predict.
 - `predictor`: a Scikit-learn predictor, e.g. a classifier, regressor or `Pipeline` object.
 - `cross_validator`: a Scikit-learn cross validator that specifies how the data should be split
   up when running the experiment.
 - `scorer` a Scikit-learn scorer that will be used as an indication of how well the classifier has
   learnt to generate predictions.

When you type `pastry run`, PyPastry does this:
 - Splits `dataset` into one or more train and test sets.
 - For each train and test set, it trains the `predictor` on the train set and generate predictions
   on the test set, and computes the score on the test set using the `scorer`.
 - Generates a results file in JSON format and stores it in a folder called `results`
 - Adds the new file and any modified files to git staging and runs a git commit.
 - Outputs the results of the experiment.

The results includes:
 - Git hash: the commit identifier from git that allows you to return to this version of the code
   at any later point in time.
 - Dataset hash: a hash generated from the dataset that will change if the dataset changes.
 - Run start: the time that the experiment run started
 - Model: the name of the `predictor` class used
 - Score: the mean ± the standard error in the mean, computed over the different folds generated
   by the `cross_validator`.
 - Duration: how long the experiment took to run, in seconds.

Contributing
------------

PyPastry is at an early stage so there's plenty to do and we'd love to have your contribution.

Check out the issues for a list of things that need doing and post a comment if you'd like to take
something on.

If you have an idea for something you'd like to do, create an issue.

Thanks for using PyPastry!
