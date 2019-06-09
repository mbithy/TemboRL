namespace TemboRL
{
    public class AgentOptions
    {
        //GLOBAL OPTIONS
        public double Gamma { get; set; }
        public double Epsilon { get; set; }
        public double Alpha { get; set; }
        //TD AGENT OPTIONS
        public string Update { get; set; }
        public bool SmoothPolicyUpdate { get; set; }
        public double Beta { get; set; }
        public int Lambda { get; set; }
        public bool ReplacingTraces { get; set; }
        public int QInitVal { get; set; }
        public int PlanN { get; set; }
        //DQN OPTIONS
        public int ExperinceAddEvery { get; set; }
        public double ExperienceSize { get; set; }
        public int LearningSteps { get; set; }
        public double ErrorClamp { get; set; }
        public int HiddenUnits { get; set; }
    }
}