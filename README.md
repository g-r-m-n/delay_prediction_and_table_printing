#Coding Challenge
Die Aufgabe sollte nicht mehr als ein paar (wenige) Stunden in Anspruch nehmen. Die Challenge besteht aus den folgenden zwei Teilaufgaben.

##Aufgabe 1: Datenexploration und Vorhersagemodell
Wir schicken dir im Nachgang einen Datensatz zu („flights.csv“), der Informationen über Flüge enthält. Bitte:
1. führe eine explorative Datenanalyse durch, insbesondere bezüglich der Frage, welche Faktoren/Treiber Verspätungen bei Ankünften (arr_delay) und Abflügen (dep_delay) beeinflussen.
1. erstelle ein Vorhersagemodell für die binäre Aussage, ob ein Abflug mehr als 30 Minuten verspätet ist (ohne Nutzung der Verspätung der Ankunft). Nutze dazu einen geeigneten Evaluationsdatensatz.


##2. Software Engineering
Bitte schreibe eine robuste, wartbare, und erweiterbare Kommandozeilenanwendung, mit der man CSV Dateien anzeigen kann.
Die Logik ist simpel: Sie ruft eine CSV Datei auf, die dann angezeigt werden soll. Bei Aufruf soll die Datei spezifiziert werden.

Die Anzeige soll folgenden Anforderungen entsprechen:
1. Jede Seite wird mit Spaltenüberschriften ausgegeben
1. Jede Seite besteht aus 10 Dateneinträgen (Zeilen)
1. Es gibt Zellenseparierungzeichen (Genau wie im Beispiel)
1. Die Spalten haben eine feste Breite, die dem längsten Eintrag je Spalte entspricht
1. Durch Drücken der Tasten F/P/N/L/E werden die im Beispiel gezeigten Befehle ausgeführt (Hierfür kannst du z.B. den input() Befehl benutzen)

So soll die Ausgabe aussehen:
Name |Age|City | -----+---+--------+ Peter|42 |New York| Paul |57 |London | Mary |35 |Munich | F)irst page, P)revious page, N)ext page, L)ast page, E)xit

Die einzelnen Seiten werden als Tabelle mit Überschrift und Zellenmarkierung dargestellt. Du kannst davon ausgegangen, dass die CSV Dateien wie folgt aussehen:
1. Die erste Zeile enthält die Überschriften
1. Die Spalten sind durch “;“ getrennt, die Kodierung ist utf-8
1. Eine neue Zeile ist durch einen Zeilenumbruch codiert, andere Zeilenumbrüche gibt es nicht
