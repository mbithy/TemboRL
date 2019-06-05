using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;

namespace TemboRL
{
    public class TD
    {
        // QAgent uses TD (Q-Learning, SARSA)
        // - does not require environment model :)
        // - learns from experience :)
        public AgentOptions Options { get; private set; }
        public int NS { get; private set; }
        public int NA { get; private set; }
        public double[] P { get; set; }
        public double[] E { get; set; }
        public double[] Q { get; set; }
        public double[] EnvModelS { get; set; }
        public double[] EnvModelR { get; set; }
        private double[] SaSeen { get; set; }
        private double[] PQ { get; set; }
        private bool Explored { get; set; }
        private int r0;
        private int s0;
        private int s1;
        private int a0;
        private int a1;
        public TD(int ns, int na, AgentOptions options)
        {
            NS = ns;
            NA = na;
            Options = options;
            Reset();
        }

        public void Reset()
        {
            // reset the agent"s policy and value function
            Q = CN.ArrayOfZeros(NS, NA);
            if (Options.QInitVal != 0)
            {
                CN.SetConst(Q, Options.QInitVal);
            }
            P = CN.ArrayOfZeros(NS * NA);
            E = CN.ArrayOfZeros(NS * NA);
            // model/planning vars
            EnvModelS = CN.ArrayOfZeros(NS * NA);
            CN.SetConst(EnvModelS, -1); // init to -1 so we can test if we saw the state before
            EnvModelR = CN.ArrayOfZeros(NS * NA);
            SaSeen = new int[] { };
            PQ = CN.ArrayOfZeros(NS * NA);
            // initialize uniform random policy
            for (var s = 0; s < NS; s++)
            {
                var poss = AllowedActions(s);
                for (var i = 0, n = poss.Length; i < n; i++)
                {
                    P[poss[i] * NS + s] = 1.0 / poss.Length;
                }
            }
            // agent memory, needed for streaming updates
            // (s0,a0,r0,s1,a1,r1,...)
            r0 = null;
            s0 = null;
            s1 = null;
            a0 = null;
            a1 = null;
        }
        public void ResetEpisode()
        {
            //773
        }
        protected int[] AllowedActions(int s)
        {

        }
        public int Act(int s)
        {
            // act according to epsilon greedy policy
            var a = 0;
            var poss = AllowedActions(s);
            var probs = new List<double>();
            for (var i = 0, n = poss.Length; i < n; i++)
            {
                probs.Add(P[poss[i] * NS + s]);
            }
            // epsilon greedy policy
            if (Math.random() < Epsilon)
            {
                var a = poss[CN.RandI(0, poss.Length)]; // random available action
                Explored = true;
            }
            else
            {
                a = poss[CN.SampleWeighted(probs.ToArray())];
                Explored = false;
            }
            // shift state memory
            s0 = s1;
            a0 = a1;
            s1 = s;
            a1 = a;
            return a;
        }
        public void Learn(int r1)
        {
            // takes reward for previous action, which came from a call to act()
            if (!(this.r0 == null))
            {
                learnFromTuple(s0, a0, r0, s1, a1, Options.Lambda);
                if (PlanN > 0)
                {
                    UpdateModel(s0, a0, r0, s1);
                    Plan();
                }
            }
            r0 = r1; // store this for next update
        }
        public void UpdateModel(int s0, int a0, int r0, int s1)
        {
            // transition (s0,a0) -> (r0,s1) was observed. Update environment model
            var sa = a0 * NS + s0;
            if (EnvModelS[sa] == -1)
            {
                // first time we see this state action
                SaSeen.Append(a0 * NS + s0); // add as seen state
            }
            EnvModelS[sa] = s1;
            EnvModelR[sa] = r0;
        }

        public void Plan()
        {
            // order the states based on current priority queue information
            var spq = new dynamic[] { };
            for (var i = 0, n = SaSeen.Length; i < n; i++)
            {
                var sa = SaSeen[i];
                var sap = PQ[sa];
                if (sap > 1e-5)
                { // gain a bit of efficiency
                    dynamic dy = new ExpandoObject();
                    dy.sa = sa;
                    dy.p = sap;
                    spq.Append(dy);
                }
            }
            var spqSorted = spq.OrderByDescending(a => a0.p).ToArray();
            spq = spqSorted;
            /*spq.sort(function (a, b) {
                return a.p < b.p ? 1 : -1
            });*/
            // perform the updates
            var nsteps = Math.Min(PlanN, spq.Length);
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
                var s1 = EnvModelS[s0a0];
                var a1 = -1; // not used for Q learning
                if (Options.Update == "sarsa")
                {
                    // generate random action?...
                    var poss = AllowedActions(s1);
                    var a1 = poss[CN.RandI(0, poss.Length)];
                }
                LearnFromTuple(s0, a0, r0, s1, a1, 0); // note Options.Lambda = 0 - shouldnt use eligibility trace here
            }
        }

        public void LearnFromTuple(int s0, int a0, int r0, int s1, int a1, int lambda)
        {
            var sa = a0 * NS + s0;
            var target = 0.0;
            // calculate the target for Q(s,a)
            if (Options.Update == "qlearn")
            {
                // Q learning target is Q(s0,a0) = r0 + gamma * max_a Q[s1,a]
                var poss = AllowedActions(s1);
                var qmax = 0;
                for (var i = 0, n = poss.Length; i < n; i++)
                {
                    var s1a = poss[i] * NS + s1;
                    var qval = Q[s1a];
                    if (i == 0 || qval > qmax)
                    {
                        qmax = qval;
                    }
                }
                target = r0 + Options.Gamma * qmax;
            }
            else if (Options.Update == "sarsa")
            {
                // SARSA target is Q(s0,a0) = r0 + gamma * Q[s1,a1]
                var s1a1 = a1 * NS + s1;
                target = r0 + Options.Gamma * Q[s1a1];
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
                var state_update = CN.ArrayOfZeros(NS);
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
                    E = zeros(NS * NA);
                }
            }
            else
            {
                // simpler and faster update without eligibility trace
                // update Q[sa] towards it with some step size
                var update = Options.Alpha * (target - Q[sa]);
                Q[sa] += update;
                UpdatePriority(s0, a0, update);
                // update the policy to reflect the change (if appropriate)
                UpdatePolicy(s0);
            }
        }

        public void UpdatePriority(int s, int a, double u)
        {
            // used in planning. Invoked when Q[sa] += update
            // we should find all states that lead to (s,a) and upgrade their priority
            // of being update in the next planning step
            u = Math.Abs(u);
            if (u < 1e-5)
            {
                return;
            } // for efficiency skip small updates
            if (PlanN == 0)
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

        public void UpdatePolicy(int s)
        {
            var poss = AllowedActions(s);
            // set policy at s to be the action that achieves max_a Q(s,a)
            // first find the maxy Q values
            var qmax, nmax;
            var qs = new double[] { };
            for (var i = 0, n = poss.length; i < n; i++)
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
            for (var i = 0, n = poss.Length; i < n; i++)
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
                for (var i = 0, n = poss.Length; i < n; i++)
                {
                    var a = poss[i];
                    P[a * NS + s] /= psum;
                }
            }
        }
    }

}

public class AgentOptions
{
    public string Update { get; set; }
    public double Gamma { get; set; }
    public double Epsilon { get; set; }
    public double Alpha { get; set; }
    public bool SmoothPolicyUpdate { get; set; }
    public double Beta { get; set; }
    public int Lambda { get; set; }
    public bool ReplacingTraces { get; set; }
    public int QInitVal { get; set; }
    public int PlanN { get; set; }
}
}