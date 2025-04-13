using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Regression_analysis
{
    public class ProgressBarConsole
    {
        public int CountIteration { get; set; }
        public int BarWidth { get; set; } = 20;

        static ConsoleColor GetProgressColor(double percent)
        {
            // Плавный переход от красного к желтому к зеленому
            if (percent < 0.5)
                return ConsoleColor.DarkRed;
            else if (percent < 0.95)
                return ConsoleColor.DarkYellow;
            else
                return ConsoleColor.DarkGreen;
        }

        public void Add(int numiteration, string head, string title) 
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            if (numiteration > CountIteration || numiteration < 0) return;
            double value = (double) numiteration / CountIteration;
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write($"{head} [");
            int filledWidth = (int) Math.Round(value * BarWidth);
            Console.ForegroundColor = GetProgressColor(value);
            Console.Write(new string('█', filledWidth));
            // Не заполненная часть
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.Write(new string('░', BarWidth - filledWidth));
            Console.ResetColor();
            Console.Write(FormattableString.Invariant($"] {value * 100 :.0}% - {title}"));

        }
    }
}
