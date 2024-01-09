from strategies.evaluation.evaluation_strategy import Evaluation_Strategy

import math
import torch
import myTransformer.MyUtils as MyUtils
from data_preparation import get_prediction_loader, load_prediciton_data, load_test_eos_data
from trim_process import EventData_Trim

MODEL_PATH = "trainedModels\\transformer50default.pth.tar"
LONGEST_TEST_ACTION_SEQUENCE = 20
ALPHA = .3
LIST_OF_BETA_VALUES = [0.1, 0.2, 0.3, 0.5]
BREAKFAST_FRAME_RATE = 15

class Longterm_Strategy(Evaluation_Strategy):
    def evaluate(self, model, optimizer, scheduler, opt):
        trim_event_data, num_types = self._lt_predict(model, optimizer, scheduler, opt)
        self._eval_prediction(trim_event_data, num_types)

    def _lt_predict(self, model, optimizer, scheduler, opt):
        print("Starting long-term prediction:")
        pred_data, num_types = load_prediciton_data(opt)
        eos_test_ti = load_test_eos_data(opt)

        # trim_event_data encapsules trimmed down versions of action sequences contained in the test.pkl
        # during this prediction, the eventdata ought to be concatenated with predictions by the model
        trim_event_data = EventData_Trim(pred_data, eos_test_ti, opt, ALPHA)
        
        pred_time = []
        pred_time_gap = []
        pred_event_type = []
        pred_event_goal = []

        predicting = True
        model.eval()
        with torch.no_grad():
            i = 0
            while(predicting):
                print("step:", i)
                i += 1
                # add the gap times to the last time when concatting
                trim_event_data.concat_predictions(pred_time_gap, pred_event_type, pred_event_goal)
                predictionloader = get_prediction_loader(opt, trim_event_data)
                pred_time = []
                pred_time_gap = []
                pred_event_type = []
                pred_event_goal = []
                ## for debugging purposes
                predicting = False
                j = 0
                
                for batch in predictionloader:
                    event_time, time_gap, event_type, event_goal, trim_time, _ , trim_type, trim_goal = map(lambda x: x.to(opt.device), batch)
                    # find out if a sequence ended
                    sequence_ended = [False] * len(trim_type)
                    for seq in range(len(trim_type)):
                        if 1 in trim_type[seq][1:]:
                            sequence_ended[seq] = True
                        else:
                            # once a sequence is found that has not yet ended, flag is set and predicition keeps on looping
                            predicting = True

                    enc_out, prediction = model(trim_type, trim_time)
                    
                    # get next event predicitons  
                    pred_types, all_types  = MyUtils.get_next_type_prediction(prediction[0], trim_type)
                    # event times
                    pred_times, all_times = MyUtils.get_next_time_prediction(prediction[1], trim_time)
                    # goals
                    pred_goals = MyUtils.get_next_goal_prediciton(prediction[2],trim_goal)
                    
                    # whenever a sequence in that batch ended, swap prediction with a zero
                    for seq in range(len(trim_type)):
                        if(sequence_ended[seq]):
                            pred_types[seq] = torch.tensor(0)
                            pred_times[seq] = torch.tensor(0)
                            pred_goals[seq] = torch.tensor(0)

                    pred_event_type += [element.item() for element in pred_types]
                    pred_time_gap += [element.item() for element in pred_times]
                    pred_event_goal += [element.item() for element in pred_goals]
                    j +=1
                # LONGEST_TEST_ACTION_SEQUENCE
                if(i > LONGEST_TEST_ACTION_SEQUENCE):
                    predicting = False
            return trim_event_data, num_types

    def _eval_prediction(self,trim_event_data, num_types):
        print("Start evaluation:")
        for beta in range(len(LIST_OF_BETA_VALUES)):
            # the predicitons
            pred_time, pred_time_gap, pred_type, pred_goal = trim_event_data.get_trim_data()
            # the ground truth
            truths_time, truths_time_gap, truths_type, truths_goal = trim_event_data.get_truths()

            num_correct = [0] * num_types
            num_total = [0] * num_types

            num_skips = 0
            for i in range(len(pred_type)):
                if any(x < 0 for x in pred_time_gap[i]):
                    num_skips += 1
                    continue
                #convert to frames
                truths_framedata = MyUtils.truth_to_frame_data(truths_time[i],truths_type[i])
                preds_framedata = MyUtils.truth_to_frame_data(pred_time[i],pred_type[i])
                # first frame of the prediciton
                alpha_index = math.ceil(len(truths_framedata) * ALPHA) + 1
                # first frame outside of prediciton window
                beta_index = math.ceil(len(truths_framedata) * (ALPHA + LIST_OF_BETA_VALUES[beta])) + 1
                len_dif = len(truths_framedata) - len( preds_framedata)
                if(len_dif > 0):
                    preds_framedata.extend([preds_framedata[-1]] * len_dif)

                tru =  truths_framedata[alpha_index:beta_index]
                pred = preds_framedata[alpha_index:beta_index]

                zips = zip(tru, pred)
                for pair in zips:
                    if(pair[0] == pair[1]):
                        num_correct[pair[0] - 1] += 1
                    num_total[pair[0] - 1] += 1

            MOC_list = [-1] * num_types 
            for index in range(num_types):
                if(num_total[index] != 0):
                    MOC_list[index] = (num_correct[index]/num_total[index]) * 100
            print(LIST_OF_BETA_VALUES[beta], MOC_list)
            count = 0
            percents = 0
            for entry in MOC_list:
                if(entry != -1):
                    count += 1
                    percents += entry
            print(f"Mean over all classes{LIST_OF_BETA_VALUES[beta]}:", percents/count)
            if(num_skips != 0):
                print("Skipped:" , num_skips)
        