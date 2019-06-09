using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TemboRL
{
    public class TDForexE : TD
    {
        //a collection of known states learned from the environment
        public Dictionary<int, int[]> KnownStates { get; set; }
        private List<double[]> Cache = new List<double[]>();
        public double CacheSize = 200;
        public TDForexE(int ns, int na, AgentOptions options) : base(ns, na, options)
        {
            NS = ns;
            NA = na;
            Options = options;            
            KnownStates = new Dictionary<int, int[]>(ns);
           Reset();
        }
        public override int[] AllowedActions(int s)
        {
            return new int[] { 0, 1 ,2};
        }
        public override int[] AllowedActions(double[] s)
        {
            //buy or wait
            if (s[5] >= -10)
            {
                return new int[] {1, 2 };
            }
            //sell or wait
            if (s[5] <= -90)
            {
                return new int[] { 0, 2 };
            }
            //whole buffet
            return new int[] { 0, 1,2 };
        }

        private void AddState(KeyValuePair<int, int[]> vps)
        {
            if (!StateExists(vps.Value))
            {
                KnownStates.Add(vps.Key, vps.Value);
            }
        }
        public void AddState(double[] positionFeature)
        {
            Cache.Add(positionFeature);
            if (Cache.Count > CacheSize)
            {
                //forget the oldest
                Cache.RemoveAt(0);
            }
            AddState(PositionFeatureToState(positionFeature));
        }

        private KeyValuePair<int, int[]> PositionFeatureToState(double[] positionFeature)
        {
            var state = new int[8];
            state[0] = positionFeature[0].ToInt();
            state[1] = positionFeature[1] >= 30 ? 1 : 0;
            //state[2] = positionFeature[2].ToInt();            
            var avg = Cache.Average(d => d[2]);
            state[2] = positionFeature[2] >= avg ? 1 : 0;
            state[3] = positionFeature[3].ToInt();
            state[4] = positionFeature[4].ToInt();
            state[5] = positionFeature[5] >= -10 ? 0 : positionFeature[6] <= -90 ? 1 : 2;
            state[6] = positionFeature[6] >= 32 ? 1 : 0;
            var fg = new KeyValuePair<int, int[]>((KnownStates.Count + 1), state);
            return fg;
        }

        public int SState(KeyValuePair<int, int[]> vps)
        {
            foreach (var x in KnownStates)
            {
                if (StateExists(vps.Value))
                {
                    return x.Key;
                }
            }
            //default, start
            return 0;
        }
        public override int StateKey(double[] positionFeature)
        {
            return SState(PositionFeatureToState(positionFeature));
        }
        private bool StateExists(int[] state)
        {
            foreach (var x in KnownStates)
            {
                if (ValuesEqual(x.Value, state) )
                {
                    return true;
                }
            }
            return false;
        }

        private bool ValuesEqual(double[] a, double[] b)
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
