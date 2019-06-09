using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using TemboRL;

namespace TRLExamples
{
    public class TDExample : TD
    {
        //a collection of known values
        public Dictionary<string,int[]> KnownStates{ get; set; }
        private List<double[]> Cache = new List<double[]>();
        public TDExample(int ns, int na, AgentOptions options) : base(ns, na, options)
        {
            NS = ns;
            NA = na;
            Options = options;
            Reset();
        }
        protected new int[] AllowedActions(int s)
        {
            return new int[] {0,1 };
        }

        public void AddState(KeyValuePair<string,int[]> vps)
        {
            if (!StateExists(vps.Value))
            {
                KnownStates.Add(vps.Key,vps.Value);
            }
        }
        public void AddState(double [] positionFeature)
        {
            Cache.Add(positionFeature);
            var state = new int[8];
            state[0] = positionFeature[0].ToInt();
            state[1] = positionFeature[1] >= 30 ? 1 : 0;
            state[2] = positionFeature[2].ToInt();
            var avg = Cache.Average(d => d[3]);
            state[3] = positionFeature[3] >= avg ? 1 : 0;
        }

        public int SState(KeyValuePair<string, int[]> vps)
        {
            foreach (var x in KnownStates)
            {
                if (StateExists(vps.Value))
                {
                    return int.Parse(x.Key);
                }
            }
            return -1;
        }
        private bool StateExists(int[] state)
        {
            foreach(var x in KnownStates)
            {
                if (ValuesEqual(x.Value, state))
                {
                    return true;
                }
            }
            return false;
        }

        private bool ValuesEqual(double[] a, double[] b)
        {
            var max = Math.Min(a.Length,b.Length);
            for(int i = 0; i < max; i++)
            {
                if (a[i] != b[i])
                {
                    return false;
                }
            }
            return true;
        }
        private bool ValuesEqual(int[] a, int[] b)
        {
            var max = Math.Min(a.Length, b.Length);
            for (int i = 0; i < max; i++)
            {
                if (a[i] != b[i])
                {
                    return false;
                }
            }
            return true;
        }


    }
}
