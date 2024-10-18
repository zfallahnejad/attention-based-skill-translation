import os

results_path = "D:/Sharif/Research_Projects/astm/evaluate/EvaluationResults"
for collection in ['java', 'php']:
    for scoring_method in ['voteshare_nobari', 'without_voteshare']:
        for result in os.listdir(results_path):
            if not result.startswith("astm"):
                continue
            if collection not in result:
                continue
            if scoring_method not in result:
                continue
            model, _ = result.split('_top')
            top_translations = _.split('_')[0]
            metrics = {'MAP': 0, 'P_1': 0, 'P_5': 0, 'P_10': 0}
            for rfile in os.listdir(results_path + '/' + result):
                if 'MAP' in rfile:
                    metric = 'MAP'
                elif 'P_10' in rfile:
                    metric = 'P_10'
                elif 'P_5' in rfile:
                    metric = 'P_5'
                else:
                    metric = 'P_1'
                with open(results_path + '/' + result + '/' + rfile) as infile:
                    numbers = []
                    for line in infile:
                        metric_value = line.strip().split(',')[-1]
                        if metric_value == "NaN":
                            numbers.append(0.0)
                        else:
                            numbers.append(float(metric_value))
                    avg_metric = sum(numbers) / len(numbers)
                    metrics[metric] = avg_metric
            print(model, top_translations, scoring_method, metrics['MAP'], metrics['P_1'], metrics['P_5'],
                  metrics['P_10'])
