using System;
using System.Collections.Generic;
using System.Text;

namespace TemboRL
{
    public class Experience
    {
        public Matrix PreviousState { get; set; }
        public int PreviousStateInt { get; set; }
        public int PreviousAction { get; set; }
        public double PreviousReward { get; set; }
        public Matrix CurrentState { get; set; }
        public int CurrentStateInt { get; set; }
        public int CurrentAction { get; set; }
    }
}
