# TemboRL
Dynamic agents for forex and BO markets

Dive in like

```C#
            var options = new AgentOptions
            {
                Gamma = 0.8,
                Epsilon = 0.25,
                Alpha = 0.8,
                ExperinceAddEvery = 5,
                ExperienceSize = 0,//infinity!
                LearningSteps = 999,
                HiddenUnits = 5000,
                ErrorClamp = 1.0,
                AdaptiveLearningSteps = true
            };            
            var temboDQN = new TemboDQN("EURUSD", 3, 2, options, true);
             var state = new double[] { 1.258977, 45.36, -65 };
            var action = temboDQN.Direction(state);
            if (action == 0)
            {
                //do buy
                //on win
                temboDQN.Reward(0.01);
                //on loss
                temboDQN.Reward(-0.05);
            }
            else
            {
                //do sell
                //on win
                temboDQN.Reward(0.01);
                //on loss
                temboDQN.Reward(-0.05);
            }
```
