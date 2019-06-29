using System;
using System.Collections.Generic;
using System.Text;

namespace TRLExamples
{
    public class Tick
    {
        public string Symbol { get; set; }

        public DateTime DateTime { get; set; }

        public double BidPrice { get; set; }

        public double AskPrice { get; set; }

        public double Quote { get; private set; }

        public void CalculateQuote()
        {
            Quote = Math.Round((AskPrice + BidPrice) / 2.0, 6);
        }

        public void SetQuote(double quote)
        {
            Quote = quote;
        }
    }
}
