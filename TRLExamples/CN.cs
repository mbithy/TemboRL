using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace TRLExamples
{
    public static class CN
    {
        private static Int64 inmemoryId;
        public static readonly string Base = AppDomain.CurrentDomain.BaseDirectory;
        public static Int64 InMemoryId()
        {
            inmemoryId += 1;
            return inmemoryId;
        }
        /// <summary>
        /// app-root\logs\log.txt
        /// </summary>
        /// <param name="logFileName"></param>
        /// <returns></returns>
        public static string LogFile(string logFileName = "log.txt")
        {
            var logs = $"{Base}logs\\";
            Directory.CreateDirectory(logs);
            return Path.Combine(logs, logFileName);
        }
        public static Object FileLock = new Object();
        /// <summary>
        /// Prints message to screen
        /// </summary>
        /// <param name="message"></param>
        /// <param name="type">0 White, 1 Red, 2 Blue, 3 Green, 4 Yellow</param>
        public static void Log(this string message, int type = 0, bool skipLog = false)
        {
            if (type == 0)
            {
                Console.ForegroundColor = ConsoleColor.White;
            }
            if (type == 1)
            {
                Console.ForegroundColor = ConsoleColor.Red;
            }
            if (type == 2)
            {
                Console.ForegroundColor = ConsoleColor.Blue;

            }
            if (type == 3)
            {
                Console.ForegroundColor = ConsoleColor.Green;
            }
            if (type == 4)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
            }
            if (type == 5)
            {
                Console.ForegroundColor = ConsoleColor.Magenta;
                Console.WriteLine(message);
                return;
            }

            Console.WriteLine(message);
            Console.ForegroundColor = ConsoleColor.White;
            if (skipLog) return;
            lock (FileLock)
            {
                File.AppendAllText(LogFile(), $"Time: {DateTime.Now.ToString("dd/MM/yyyy hh:mm:ss")} {message}\r\n");
            }
        }
    }
}
