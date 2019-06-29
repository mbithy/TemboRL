using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using TemboRL;
using TemboRL.Agent;
using TemboRL.Models;

namespace TRLExamples
{
    public class TemboDQN
    {
        public DQN dqnAgent;
        public bool IsDone = false;
        private List<Candlestick> Candlesticks;
        public Dictionary<string, State> Historical { get; set; }
        public double WinRate { get; set; }
        public double RewardCollected = 0;
        public bool BackgroundLearning { get; set; }
        public string AgentName { get; set; }
        private static Thread ForeverLearner;
        private static int lastAction;
        private static string lastKey;
        public int total, wins;
        public TemboDQN(string agentName, int ns, int na, AgentOptions options, bool backgroundLearning = false)
        {
            AgentName = agentName;
            BackgroundLearning = backgroundLearning;
            dqnAgent = new DQN(ns, na, options);
            Candlesticks = new List<Candlestick>();
            Historical = new Dictionary<string, State>();
            ForeverLearner = new Thread(BLearning)
            {
                IsBackground = true,
                Priority = ThreadPriority.BelowNormal//just incase this agent collective is plugged to a bigger collective running threads of a higher priority
            };
            if (BackgroundLearning)
            {
                ForeverLearner.Start();
            }
        }

        public void StartLearning()
        {
            ForeverLearner.Start();
        }

        private void BLearning()
        {
            while (true)
            {
                if (Historical.Count < 20000)
                {
                    //
                    Thread.Sleep(TimeSpan.FromMinutes(30));
                }
                var correct = 0.0;
                var total = 0.0;
                var options = new AgentOptions
                {
                    Gamma = Tembo.Random(0.01, 0.99),
                    Epsilon = Tembo.Random(0.01, 0.75),
                    Alpha = Tembo.Random(0.01, 0.99),
                    ExperinceAddEvery = Tembo.RandomInt(1, 10000),
                    ExperienceSize = 0,
                    LearningSteps = Tembo.RandomInt(1, 10),
                    HiddenUnits = Tembo.RandomInt(100000, 100000000),
                    ErrorClamp = Tembo.Random(0.01, 1.0),
                    AdaptiveLearningSteps = true
                };
                var agent = new DQN(dqnAgent.NumberOfStates, dqnAgent.NumberOfActions, options);
                for (var i = 0; i < Historical.Count; i++)
                {
                    var spi = Historical.ElementAt(i);
                    var action = agent.Act(spi.Value.Values);
                    if (action == spi.Value.Output)
                    {
                        correct += 1;
                        agent.Learn(1);
                    }
                    else
                    {
                        agent.Learn(-1);
                    }
                    total += 1;
                }
                var winrate = (correct / total) * 100;
                if (winrate > WinRate)
                {
                    CN.Log($"NEW AGENT DISCOVERED --> WINRATE {winrate.ToString("p")}, CLASS: {AgentName}", 2);
                    Save();
                    dqnAgent = agent;
                    WinRate = winrate;
                }
            }
        }

        public void Save()
        {
            var path = $"{CN.Base}Agents\\{AgentName}.temboai";
            File.WriteAllText(path, this.ToJson());
        }

        public int Direction(double[] state)
        {
            var sta = new State
            {
                Values = state,
                Occurrence = 1,
                Output = -1
            };
            lastKey = Tembo.GetId();
            Historical.Add(lastKey, sta);
            lastAction = dqnAgent.Act(state);
            return lastAction;
        }

        private bool StateExists(double[] state)
        {
            foreach (var x in Historical)
            {
                if (x.Value.EqualsTo(state))
                {
                    x.Value.Occurrence++;
                    lastKey = x.Key;
                    return true;
                }
            }
            return false;
        }

        public void Reward(double value)
        {
            total += 1;
            if (value > 0)
            {
                wins += 1;
            }
            WinRate = wins / total;
            Historical[lastKey].Output = value < 0 ? lastAction == 1 ? 0 : 1 : lastAction;
            RewardCollected += value;
            dqnAgent.Learn(value);
        }
        public void Reward(double value, int outPut)
        {
            total += 1;
            if (value > 0)
            {
                wins += 1;
            }
            var last = Historical.LastOrDefault();
            Historical[last.Key].Output = outPut;
            WinRate = wins / total;
            RewardCollected += value;
            dqnAgent.Learn(value);
        }
    }
}
