using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;

namespace TemboRL
{
    public abstract class TD
    {
        // QAgent uses TD (Q-Learning, SARSA)
        // - does not require environment model :)
        // - learns from experience :)
        public AgentOptions Options { get; set; }
        public int NS { get; set; }
        public int NA { get;set; }
        public double[] P { get; set; }
        public double[] E { get; set; }
        public double[] Q { get; set; }
        public double[] EnvModelS { get; set; }
        public double[] EnvModelR { get; set; }
        private double[] SaSeen { get; set; }
        private double[] PQ { get; set; }
        //public int [] AllowedActions { get; set; }
        private bool Explored { get; set; }
        private double r0;
        private int s0;
        private int s1;
        private int a0;
        private int a1;
        public TD(int ns, int na, AgentOptions options)
        {
            NS = ns;
            NA = na;
            Options = options;
            //AllowedActions = allowedActions;
            Reset();
        }

        protected void Reset()
        {
            // reset the agent"s policy and value function
            Q = CM.ArrayOfZeros(NS* NA);
            if (Options.QInitVal != 0)
            {
                CM.SetConst(Q, Options.QInitVal);
            }
            P = CM.ArrayOfZeros(NS * NA);
            E = CM.ArrayOfZeros(NS * NA);
            // model/planning vars
            EnvModelS = CM.ArrayOfZeros(NS * NA);
            CM.SetConst(EnvModelS, -1); // init to -1 so we can test if we saw the state before
            EnvModelR = CM.ArrayOfZeros(NS * NA);
            SaSeen = new double[] { };
            PQ = CM.ArrayOfZeros(NS * NA);
            // initialize uniform random policy
            for (var s = 0; s < NS; s++)
            {
                var poss = AllowedActions(s);
                for (var i = 0;i<poss.Length; i++)
                {
                    P[poss[i] * NS + s] = 1.0 / poss.Length;
                }
            }
            // agent memory, needed for streaming updates
            // (s0,a0,r0,s1,a1,r1,...)
            r0 = 999999999;
            s0 = 999999999;
            s1 = 999999999;
            a0 = 999999999;
            a1 = 999999999;
        }
        private void ResetEpisode()
        {
            //773
        }
        public virtual int[] AllowedActions(int s)
        {
            //default 0-buy, 1-sell
            return new int[] { 0,1};
        }
        public virtual int[] AllowedActions(double[] s)
        {
            //default 0-buy, 1-sell
            return new int[] { 0, 1 };
        }
        public int Act(int s)
        {
            // act according to epsilon greedy policy
            var a = 0;
            var poss = AllowedActions(s);
            var probs = new List<double>();
            for (var i = 0;i<poss.Length; i++)
            {
                probs.Add(P[poss[i] * NS + s]);
            }
            // epsilon greedy policy
            if (CM.Random() < Options.Epsilon)
            {
                a = poss[CM.RandomInt(0, poss.Length)]; // random available action
                Explored = true;
            }
            else
            {
                a = poss[CM.SampleWeighted(probs.ToArray())];
                Explored = false;
            }
            // shift state memory
            s0 = s1;
            a0 = a1;
            s1 = s;
            a1 = a;
            return a;
        }
        public int Act(double[] state)
        {
            var s = StateKey(state);
            // act according to epsilon greedy policy
            var a = 0;
            var poss = AllowedActions(state);
            var probs = new List<double>();
            for (var i = 0; i < poss.Length; i++)
            {
                probs.Add(P[poss[i] * NS + s]);
            }
            // epsilon greedy policy
            if (CM.Random() < Options.Epsilon)
            {
                a = poss[CM.RandomInt(0, poss.Length)]; // random available action
                Explored = true;
            }
            else
            {
                a = poss[CM.SampleWeighted(probs.ToArray())];
                Explored = false;
            }
            // shift state memory
            s0 = s1;
            a0 = a1;
            s1 = s;
            a1 = a;
            return a;
        }
        public virtual int StateKey(double[] positionFeature)
        {
            return 0;
        }
        public void Learn(double r1)
        {
            // takes reward for previous action, which came from a call to act()
            if (!(r0 == 999999999))
            {
                var exp = new Experience
                {
                    PreviousStateInt = s0,
                    PreviousAction = a0,
                    PreviousReward = r0,
                    CurrentStateInt = s1,
                    CurrentAction = a1
                };
                LearnFromTuple(/*s0, a0, r0, s1, a1*/exp, Options.Lambda);
                if (Options.PlanN > 0)
                {
                    UpdateModel(s0, a0, r0, s1);
                    Plan();
                }
            }
            r0 = r1; // store this for next update
        }
        private void UpdateModel(int state0, int action0, double reward0, int state1)
        {
            // transition (s0,a0) -> (r0,s1) was observed. Update environment model
            var sa = action0 * NS + state0;
            if (EnvModelS[sa] == -1)
            {
                // first time we see this state action
                SaSeen.Append(action0 * NS + state0); // add as seen state
            }
            EnvModelS[sa] = state1;
            EnvModelR[sa] = reward0;
        }

        private void Plan()
        {
            // order the states based on current priority queue information
            var spq = new List<dynamic>();
            for (var i = 0; i < SaSeen.Length; i++)
            {
                var sa = SaSeen[i].ToInt();
                var sap = PQ[sa];
                if (sap > 1e-5)
                { // gain a bit of efficiency
                    dynamic dy = new ExpandoObject();
                    dy.sa = sa;
                    dy.p = sap;
                    spq.Add(dy);
                }
            }
            var spqSorted = spq.OrderByDescending(a => a.p).ToList();
            spq = spqSorted;
            /*spq.sort(function (a, b) {
                return a.p < b.p ? 1 : -1
            });*/
            // perform the updates
            var nsteps = Math.Min(Options.PlanN, spq.Count);
            for (var k = 0; k < nsteps; k++)
            {
                // random exploration
                //var i = randi(0, SaSeen.Length); // pick random prev seen state action
                //var s0a0 = SaSeen[i];
                var s0a0 = spq[k].sa;
                PQ[s0a0] = 0; // erase priority, since we"re backing up this state
                var s0 = s0a0 % NS;
                var a0 = Math.Floor(s0a0 / NS);
                var r0 = EnvModelR[s0a0];
                var s1 = EnvModelS[s0a0].ToInt();
                var a1 = -1; // not used for Q learning
                if (Options.Update == "sarsa")
                {
                    // generate random action?...
                    var poss = AllowedActions(s1);
                    a1 = poss[CM.RandomInt(0, poss.Length)];
                }
                var exp = new Experience
                {
                    PreviousStateInt = s0,
                    PreviousAction = a0,
                    PreviousReward = r0,
                    CurrentStateInt = s1,
                    CurrentAction = a1
                };
                LearnFromTuple(exp, 0); // note Options.Lambda = 0 - shouldnt use eligibility trace here
            }
        }

        private void LearnFromTuple(Experience exp/*int s0, int a0, double r0, int s1, int a1*/, int lambda)
        {
            var sa =exp.PreviousAction * NS + exp.PreviousStateInt;
            var target = 0.0;
            // calculate the target for Q(s,a)
            if (Options.Update == "qlearn")
            {
                // Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
                var poss =  AllowedActions(exp.CurrentStateInt);
                var qmax = 0.0;
                for (var i = 0;i<poss.Length; i++)
                {
                    var s1a = poss[i] * NS + exp.CurrentStateInt;
                    var qval = Q[s1a];
                    if (i == 0 || qval > qmax)
                    {
                        qmax = qval;
                    }
                }
                target = exp.PreviousReward + Options.Gamma * qmax;
            }
            else if (Options.Update == "sarsa")
            {
                // SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
                var s1a1 =exp.CurrentAction * NS + exp.CurrentStateInt;
                target = exp.PreviousReward + Options.Gamma * Q[s1a1];
            }
            if (lambda > 0)
            {
                // perform an eligibility trace update
                if (Options.ReplacingTraces)
                {
                    E[sa] = 1;
                }
                else
                {
                    E[sa] += 1;
                }
                var edecay = lambda * Options.Gamma;
                var state_update = CM.ArrayOfZeros(NS);
                for (var s = 0; s < NS; s++)
                {
                    var poss = AllowedActions(s);
                    for (var i = 0; i < poss.Length; i++)
                    {
                        var a = poss[i];
                        var saloop = a * NS + s;
                        var esa = E[saloop];
                        var update = Options.Alpha * esa * (target - Q[saloop]);
                        Q[saloop] += update;
                        UpdatePriority(s, a, update);
                        E[saloop] *= edecay;
                        var u = Math.Abs(update);
                        if (u > state_update[s])
                        {
                            state_update[s] = u;
                        }
                    }
                }
                for (var s = 0; s < NS; s++)
                {
                    if (state_update[s] > 1e-5)
                    { // save efficiency here
                        UpdatePolicy(s);
                    }
                }
                if (Explored && Options.Update == "qlearn")
                {
                    // have to wipe the trace since q learning is off-policy :(
                    E = CM.ArrayOfZeros(NS * NA);
                }
            }
            else
            {
                // simpler and faster update without eligibility trace
                // update Q[sa] towards it with some step size
                var update = Options.Alpha * (target - Q[sa]);
                Q[sa] += update;
                UpdatePriority(exp.PreviousStateInt, exp.PreviousAction, update);
                // update the policy to reflect the change (if appropriate)
                UpdatePolicy(exp.PreviousStateInt);
            }
        }

        private void UpdatePriority(int s, int a, double u)
        {
            // used in planning. Invoked when Q[sa] += update
            // we should find all states that lead to (s,a) and upgrade their priority
            // of being update in the next planning step
            u = Math.Abs(u);
            if (u < 1e-5)
            {
                return;
            } // for efficiency skip small updates
            if (Options.PlanN == 0)
            {
                return;
            } // there is no planning to be done, skip.
            for (var si = 0; si < NS; si++)
            {
                // note we are also iterating over impossible actions at all states,
                // but this should be okay because their env_model_s should simply be -1
                // as initialized, so they will never be predicted to point to any state
                // because they will never be observed, and hence never be added to the model
                for (var ai = 0; ai < NA; ai++)
                {
                    var siai = ai * NS + si;
                    if (EnvModelS[siai] == s)
                    {
                        // this state leads to s, add it to priority queue
                        PQ[siai] += u;
                    }
                }
            }
        }

        private void UpdatePolicy(int s)
        {
            var poss = AllowedActions(s);
            // set policy at s to be the action that achieves max_a Q(s,a)
            // first find the maxy Q values
            double qmax = 0;double nmax=0;
            var qs = new double[poss.Length];
            for (var i = 0; i < poss.Length; i++)
            {
                var a = poss[i];
                var qval = this.Q[a * NS + s];
                qs.Append(qval);
                if (i == 0 || qval > qmax)
                {
                    qmax = qval;
                    nmax = 1;
                }
                else if (qval == qmax)
                {
                    nmax += 1;
                }
            }
            // now update the policy smoothly towards the argmaxy actions
            var psum = 0.0;
            for (var i = 0;i<poss.Length; i++)
            {
                var a = poss[i];
                var target = (qs[i] == qmax) ? 1.0 / nmax : 0.0;
                var ix = a * NS + s;
                if (Options.SmoothPolicyUpdate)
                {
                    // slightly hacky :p
                    P[ix] += Options.Beta * (target - P[ix]);
                    psum += P[ix];
                }
                else
                {
                    // set hard target
                    P[ix] = target;
                }
            }
            if (Options.SmoothPolicyUpdate)
            {
                // renomalize P if we're using smooth policy updates
                for (var i = 0; i < poss.Length; i++)
                {
                    var a = poss[i];
                    P[a * NS + s] /= psum;
                }
            }
        }
    }
}