#
library(ISLR2)
library(survival)
library(survminer)
#
BrainCancer |> head()
#
model_1 = coxph(
  Surv(time, status) ~ sex+diagnosis+loc+ki+gtv+stereo, data = BrainCancer
)
summary(model_1)
#
Publication |> head()
#
model_3 = coxph(
  Surv(time, status) ~ posres+multi+clinend+mech+sampsize+budget+impact, 
  data = Publication
)
summary(model3)
#

#
