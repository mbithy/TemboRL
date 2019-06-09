using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TemboRL
{
    public class DQN
    {
        public int HiddenUnits { get; }
        public int NumberOfStates { get; }
        public int NumberOfActions { get; }
        private Dictionary<string, Matrix> Network { get;}
        private List<Experience> Memory { get; }
        private Graph LastGraph { get; set; }
        public AgentOptions Options { get; }        
        private double r0, TDError;
        private Matrix s0;
        private Matrix s1;
        private int a0;
        private int a1;
        private int t;       

        public DQN(int numberOfStates, int numberOfActions, AgentOptions options)
        {
            NumberOfStates = numberOfStates;
            NumberOfActions = numberOfActions;
            Options = options;
            HiddenUnits = options.HiddenUnits;
            Network = new Dictionary<string, Matrix>
            {
                //w1
                { "W1", CM.RandomMatrix(HiddenUnits, NumberOfStates, 0, 0.01) },
                //b1
                { "B1", CM.RandomMatrix(HiddenUnits, 1, 0, 0.01) },
                //w2
                { "W2", CM.RandomMatrix(NumberOfActions, HiddenUnits, 0, 0.01) },
                //b2
                { "B2", CM.RandomMatrix(NumberOfActions, 1, 0, 0.01) }
            };
            Memory = new List<Experience>();
        }

        /// <summary>
        /// Returns an action from a state
        /// </summary>
        /// <param name="state">state size must be equal to NumberOfStates</param>
        /// <returns></returns>
        public int Act(double[] state)
        {
            CM.Assert(state.Length == NumberOfStates);
            var a = 0;
            // convert to a Mat column vector
            var s = new Matrix(NumberOfStates, 1);
            s.Set(state);

            // epsilon greedy policy
            if (CM.Random() < Options.Epsilon)
            {
                a = CM.RandomInt(0, NumberOfActions);
            }
            else
            {
                // greedy wrt Q function
                var amat = ForwardQ(Network, s, false);
                a = CM.Maxi(amat.W); // returns index of argmax action
            }

            // shift state memory
            this.s0 = this.s1;
            this.a0 = this.a1;
            this.s1 = s;
            this.a1 = a;
            return a;
        }
        /// <summary>
        /// Rewards the agent for perfomic an action
        /// ,memorizes and kearns from the experience
        /// </summary>
        /// <param name="reward"></param>
        public void Learn(double reward)
        {
            // perform an update on Q function
            if (this.r0 >0 && Options.Alpha > 0)
            {

                // learn from this tuple to get a sense of how "surprising" it is to the agent
                var exp = new Experience
                {
                    PreviousState = s0,
                    PreviousAction = a0,
                    PreviousReward = r0,
                    CurrentState = s1,
                    CurrentAction = a1
                };
                var tderror = LearnFromExperience(exp);
                TDError= tderror; // a measure of surprise

                // decide if we should keep this experience in the replay
                if (t % Options.ExperinceAddEvery== 0)
                {
                    Memory.Add(new Experience { PreviousState = s0, PreviousAction = a0, PreviousReward = r0, CurrentState = s1, CurrentAction = a1 });
                    if (Memory.Count > Options.ExperienceSize)
                    {
                        //forget oldest
                        Memory.RemoveAt(0);
                    }
                }
                this.t += 1;

                // sample some additional experience from replay memory and learn from it
                for (var k = 0; k < Options.LearningSteps; k++)
                {
                    var ri = CM.RandomInt(0, Memory.Count); // todo: priority sweeps?
                    var e = Memory[ri];
                    LearnFromExperience(e);
                }
            }
            this.r0 = reward; // store for next update
        }

        private Matrix ForwardQ(Dictionary<string, Matrix> net, Matrix s, bool needsBackProp)
        {
            var G = new Graph(needsBackProp);
            var a1mat = G.Add(G.Multiply(net["W1"], s), net["B1"]);
            var h1mat = G.Tanh(a1mat);
            var a2mat = G.Add(G.Multiply(net["W2"], h1mat), net["B2"]);
            LastGraph = G; // back this up. Kind of hacky isn't it
            return a2mat;
        }
        private double LearnFromExperience(Experience experience/*Matrix s0, int a0, double r0, Matrix s1, int a1*/)
        {
            // want: Q(s,a) = r + gamma * max_a' Q(s',a')

            // compute the target Q value
            var tmat =ForwardQ(Network, s1, false);
            var qmax = r0 + Options.Gamma * tmat.W[CM.Maxi(tmat.W)];

            // now predict
            var pred = ForwardQ(Network, s0, true);

            var tderror = pred.W[a0] - qmax;
            var clamp =Options.ErrorClamp;
            if (Math.Abs(tderror) > clamp)
            {  // huber loss to robustify
                if (tderror > clamp) tderror = clamp;
                if (tderror < -clamp) tderror = -clamp;
            }
            pred.DW[a0] = tderror;
            LastGraph.Backward(); // compute gradients on net params

            // update net
            CM.UpdateNetwork(Network, Options.Alpha);
            return tderror;
        }
    }
}
