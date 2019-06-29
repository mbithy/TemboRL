using System;
using TemboRL.Models;

namespace TRLExamples

  
{
    class Program
    {
        
        static void Main(string[] args)
        {
            //initiate with random values and let backgroud learning do the magic for you
            var options = new AgentOptions
            {
                Gamma = 0.8,
                Epsilon = 0.25,
                Alpha = 0.8,
                ExperinceAddEvery = 5,
                ExperienceSize = 0,//infinity!
                LearningSteps = 999,
                HiddenUnits = 5000000,//will take forever on your pc  get BARE METAL --> http://bit.ly/CheapVPSWin
                ErrorClamp = 1.0,
                AdaptiveLearningSteps = true
            };
            //we only have 2 options BUY or SELL
            var temboDQN = new TemboDQN("EURUSD", 3, 2, options, true);

            //get state from market
            //for example you can pass the values of 3 different indicators (don't limit yourself though you can pass the entire catalogue of indicators since 1970)
            // MA, RSI, WPR
            //categorical values like 1,0 eg signal state of indicator might get better values
            //var categoricalState = new double[] { 1, 0, 0 }; MA is buy, RSI wait, WPR wait
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
        }
    }
}
