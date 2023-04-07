import numpy as np

def accuracy(ensemble_val_predictions, true_labels):
    """ return accuracy of an ensemble 

    ensemble_val_predictions : List[ndarray]
        dim 0 : base learners 
        dim 1 : samples 
        dim 2 : class probabilities 
        example : ensemble_val_predictions[0][5][2] is the probability prediction of the first base learner for class 3 for the 6th sample
    true_labels : List[int]
        list of true labels 
    """
    prob_predictions = np.mean(ensemble_val_predictions, axis = 0)
    predictions = np.argmax(prob_predictions, axis = 1)
    correct = true_labels == predictions
    return sum(correct) / len(true_labels)
    

def greedy_selection_without_replacement(n_members, val_predictions, true_labels, metric_fn, minimize_metric_fn = False):
    """ build ensemble by sequentially selecting the model that is best for accuracy of the ensemble without replacement

    n_members : int
        how many member the final ensemble should have
    val_predictions : ndarray
        dim 0 : base learners
        dim 1 : samples
        dim 2 : class probabilities
    true_labels : List[int]
        list of true labels
    metric_fn : Callabel
        metric the ensemble seletion should optimize - must return strictly non-negative values
    minimize_metric_fn : bool
        indicates whether to minimize or maximize the metric function

    Returns
    -------
    ndarray[bool] each flag indicates if the baselearner at the corresponding index is in the array or not
    
    """
    ensemble_membership_flags = [False] * len(val_predictions) #has True for indices of base learners that end up in the ensemble
    ensemble_val_predictions = [] #stores the val_predictions of the ensemble members

    minimize_factor = -1 if minimize_metric_fn else 1 #is -1 if we want to minimize the metric

    for _ in range(n_members):
        current_best_next_index = ensemble_membership_flags.index(False)
        current_best_performance = metric_fn(ensemble_val_predictions + [val_predictions[current_best_next_index]], true_labels)

        for i in range(current_best_next_index, len(ensemble_membership_flags)):
            #for every base learner (that might not yet be in the ensemble)
            if(ensemble_membership_flags[i]):
                #skip if already in ensemble
                continue 

            performance = metric_fn(ensemble_val_predictions + [val_predictions[i]], true_labels)

            if(performance * minimize_factor > current_best_performance * minimize_factor):
                current_best_next_index = i
                current_best_performance = performance
        
        ensemble_val_predictions.append(val_predictions[current_best_next_index])
        ensemble_membership_flags[current_best_next_index] = True
    return np.array(ensemble_membership_flags)


            

