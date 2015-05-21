import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
loanData = pd.read_csv('LoanStats3.csv', header=1, low_memory=False)
loanData['issue_d_format'] = pd.to_datetime(loanData['issue_d'])
loanDataTS = loanData.set_index('issue_d_format')
year_month_summary = loanDataTS.groupby(lambda x: x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']
#print loan_count_summary
plt.figure()
p = loan_count_summary.hist()
plt.show()
sm.graphics.tsa.plot_acf(loan_count_summary)
sm.graphics.tsa.plot_pacf(loan_count_summary)
