using System;
using System.Collections.Generic;
using System.Text;

namespace TRLExamples
{
    public class State
    {
        public double[] Values { get; set; }

        public int Output { get; set; }

        public int Occurrence { get; set; }

        public bool EqualsTo(double[] state)
        {
            if (state.Length != Values.Length)
                return false;
            for (int index = 0; index < Values.Length; ++index)
            {
                if (Values[index] != state[index])
                    return false;
            }
            return true;
        }
    }
}
