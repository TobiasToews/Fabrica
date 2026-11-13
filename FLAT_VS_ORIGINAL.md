# Flat-Pipeline im Vergleich zum ursprünglichen Fabrica-Planer

Stand: 13.11.2026. Vergleichsbasis ist das offizielle Fabrica-Release (Commit `215a30f`, `README.md`,
`planning/run_planning.sh`). Die Flat-Pipeline liegt im Arbeitsverzeichnis und ist noch nicht committet.
Bedienung und alle Einzelwerte stehen in `FLAT_PIPELINE.md`, die Läufe je Baugruppe in `assembly_runs/`.

## 1. Kurzfassung

- **Original:** plant eine Montage mit zwei Roboterarmen. Die Teile stecken in einem gedruckten Fixture, das jedes Teil so
  hält, dass der gewählte Griff senkrecht von oben kommt. Der Plan wird nicht gegen die Grenzen des realen Roboters geprüft.
- **Flat:** kein Fixture. Die Teile liegen flach auf dem Tisch, jedes in einer Lage, in der es von selbst liegen bleibt.
  Ein realer Arm (der Bewegungsarm) nimmt und fügt alle Teile, ein Mensch hält das erste Teil. Jeder Pfad des realen Arms
  wird gegen feste Regeln geprüft; ein Plan gilt erst als fertig, wenn die Schlussprüfung besteht.
- **Der Kernunterschied:** Im Original richtet sich die Lage des Teils nach dem Griff (das Fixture wird passend gebaut).
  Ohne Fixture ist es umgekehrt: Die Lage des Teils steht fest, und ein Griff muss zu zwei Lagen passen, zur liegenden
  (Aufnehmen) und zur montierten (Einsetzen). Daraus folgt der größte Teil der Änderungen.

## 2. Der ursprüngliche Planer

Sechs Stufen, gestartet mit `bash planning/run_planning.sh EXP_NAME ASSEMBLY`:

| # | Stufe | Skript | Was sie tut |
|---|---|---|---|
| 1 | Precedence planning | `run_preced_plan.py` | Zerlegt die Baugruppe in der Simulation: Welches Teil muss vor welchem da sein, und auf welchem Pfad wird jedes Teil eingesetzt. |
| 2 | Grasp planning | `run_grasp_arm_gen.py` | Griffe je Teil in der montierten Lage: gegenüberliegende Punktpaare auf der Oberfläche (200 Punkte), 10 Handrichtungen je Paar. Ein Griff bleibt, wenn der Arm ihn entlang des Einschubpfads erreicht und nichts kollidiert. Von den brauchbaren werden 100 zufällig behalten, getrennt für Bewegungs- und Haltearm. |
| 3 | Sequence planning | `run_seq_plan.py` | Baum aller machbaren Reihenfolgen mit passenden Griffpaaren (ein Arm hält, einer fügt). |
| 4 | Sequence optimization | `run_seq_opt.py` | Wählt Reihenfolge und Griffe: möglichst viele stabile Haltegriffe, wenige Griffwechsel, kleines Drehmoment der Einschubkräfte. |
| 5 | Fixture generation | `run_fixture_gen.py` | Dreht jedes Teil so, dass sein gewählter Griff senkrecht von oben kommt, packt die Teile auf eine Fläche von höchstens 20 x 25 cm (ein Druck) oder 20 x 50 cm (zwei Drucke) und erzeugt daraus das Fixture für den 3D-Druck. |
| 6 | Arm motion planning | `run_motion_plan.py` | Pfade beider Arme (RRT-Connect, danach Glättung). Kollisionen werden mit 0,5 cm Puffer um die Teile geprüft. |

Dieser Ablauf existiert unverändert weiter. Die Flat-Pipeline ist ein zweiter Einstieg (`planning/run_planning_flat.py`),
und die Griff-Erzeugung verhält sich ohne den Schalter `--flat` wie im Original.

## 3. Die Unterschiede auf einen Blick

| Thema | Original | Flat-Pipeline |
|---|---|---|
| Ausgangslage der Teile | im gedruckten Fixture, jedes in der Lage, die zum Griff passt | flach auf dem Tisch, in einer stabilen Ruhelage; statt des Fixtures eine 1:1-Schablone zum Ausdrucken |
| Aufnehmen | Hand immer senkrecht von oben | Hand so, wie Griff und Ruhelage es ergeben, auch geneigt; Greifer mindestens 0,5 cm, Handgelenk 1 cm über dem Tisch |
| Roboter | zwei Arme (Bewegungs- und Haltearm), beide geplant | ein realer Arm, alle seine Pfade geprüft; der Haltearm ist ein Mensch (`--single-arm`), seine Pfade werden nur angezeigt |
| Erstes Teil | der Haltearm holt es aus dem Fixture | der Mensch stellt es auf die Schablone; kann der Haltegriff das liegende Teil nicht aufnehmen, bringt es der Bewegungsarm als geplanten Schritt an seinen Platz |
| Auswahl der Griffe | 100 zufällig aus allen, die montiert funktionieren | nur Griffe, die auch das liegende Teil aufnehmen können, nach Bewertung ausgewählt |
| Fingerkontakt | 200 Zufallspunkte auf dem ganzen Finger | festes Raster (0,25 mm) auf den beiden Greifflächen; verlangt werden 3 mm Fingertiefe am schwächeren Finger |
| Platz der Teile | vom Fixture vorgegeben | Suche Teil für Teil nach einem freien Platz, an dem die Aufnahme alle Prüfungen besteht |
| Bewegungsplanung | freier Pfad zwischen Start und Ziel mit 0,5 cm Puffer | zusätzlich gerade Stücke in jeden Griff hinein und heraus; es wird mit denselben Regeln geplant, nach denen später geprüft wird |
| Grenzen des realen Roboters | kommen nicht vor | Roboterhöhe unter dem Tisch, virtuelle Wände von franka-interface, Abstand zu Singularitäten und Gelenkgrenzen |
| Prüfung des fertigen Plans | keine | jede Stellung jedes Pfads wird mit echter Geometrie nachgemessen (`plan_checks.py`), danach Schlussprüfung (`verify_plan.py`) |
| Wenn etwas nicht geht | das Skript bricht ab | die fehlschlagende Stufe nennt die Ursache; eine Schleife verwirft daraufhin den Griff oder den Ablageplatz und plant neu (bis zu 10 Versuche) |
| Platz von Baugruppe und Teilen | fest | frei wählbar und im Log gespeichert; ein Positions-Test prüft viele Plätze |
| Tests des Planers | keine | 90 Tests der Regeln, des Planers und der Griff-Funktionen |

## 4. Stufe für Stufe

| # | Original | Flat-Pipeline | Was sich ändert |
|---|---|---|---|
| 1 | `run_preced_plan.py` | dasselbe Skript | nichts an der Planung |
| 2 | `run_grasp_arm_gen.py` | `run_grasp_arm_gen.py --flat` | Griffe müssen zur liegenden und zur montierten Lage passen (5.2). Hat ein einzufügendes Teil keinen Griff, endet die Planung hier mit einer Meldung. |
| 2b | gibt es nicht | `run_pickup_gen.py --filter` | wendet dieselben Kriterien noch einmal auf `grasps.pkl` an und hält die Einstellungen fest |
| 3 | `run_seq_plan.py` | dasselbe Skript | nichts an der Planung |
| 4 | `run_seq_opt.py` | `run_pickup_gen.py --seq-opt` | dieselbe Optimierung mit einer Zusatzbedingung: Der erste Haltegriff muss das liegende Teil aufnehmen können, sonst greift der Rückfall für das erste Teil |
| 5 | `run_fixture_gen.py` (Fixture) | `run_pickup_gen.py` (Ablage) | ersetzt: statt eines Fixtures eine geprüfte Ablage der liegenden Teile mit Schablone (5.3) |
| 6 | `run_motion_plan.py` | `run_motion_plan.py --optimized --no-viz --single-arm` | strenger Planer für den realen Arm, jeder Pfad geprüft und bei Bedarf neu geplant (5.4, 5.5) |
| 7 | gibt es nicht | Schleife in `run_planning_flat.py` | reagiert auf die Ursache eines Fehlschlags (5.6) |
| 8 | gibt es nicht | `verify_plan.py`, `workcell.json` | Schlussprüfung; die realen Werte, für die der Plan gilt, werden mit dem Plan gespeichert |

## 5. Die neuen Bausteine

### 5.1 Ruhelagen (`flat_grasp.get_flat_rots`)

Im Original gibt es den Begriff nicht, weil das Fixture jede Lage halten kann. Ohne Fixture muss das Teil von selbst
liegen bleiben. Ein Teil kann auf jeder Fläche seiner konvexen Hülle liegen, über der sein Schwerpunkt steht. Der
Kippwinkel sagt, wie weit man es neigen kann, bevor es umfällt: ab 20° gilt eine Lage als stabil; Lagen zwischen 10° und
20° werden nur benutzt, wenn kein Griff zu einer stabilen passt, und auf der Schablone markiert. Gerechnet wird am zum
Ursprung verschobenen Teil, damit das Ergebnis nicht davon abhängt, wo die Baugruppe steht.

### 5.2 Griffe für zwei Lagen (`run_grasp_arm_gen.py --flat`, `flat_grasp.py`)

| Schritt | Original | Flat |
|---|---|---|
| Kandidaten | Punktpaare am montierten Teil | dieselben, und zusätzlich Kandidaten, die am liegenden Teil erzeugt werden (Hand möglichst von oben, Finger so tief wie der Tisch erlaubt); jede Sorte wird an der jeweils anderen Lage geprüft |
| brauchbar | erreichbar und kollisionsfrei in der montierten Lage | zusätzlich: nimmt das liegende Teil auf, bleibt in den virtuellen Wänden, weg von Singularitäten und Gelenkgrenzen, 3 mm Fingertiefe |
| gerade Bewegungen an der Baugruppe | nicht geprüft | der Arm muss das Teil gerade zum Einschub-Startpunkt bringen und nach dem Loslassen gerade zurückfahren können; sonst fällt der Griff weg |
| zu wenige Griffe | bleibt so | zweite Runde mit 600 weiteren Punktpaaren, wenn ein Teil weniger als 10 brauchbare hat |
| Auswahl | 100 zufällig | die 100 am besten bewerteten, verteilt über die Anfahrrichtungen. Bewertung: Griffsicherheit 0,4; Schwerpunkt nahe der Fingerlinie 0,25; aufrechte Hand 0,2; Einschub-Bewertung des Originals 0,15 |
| Haltegriffe | 100 zufällig | wie im Original, ergänzt um Haltegriffe, die das erste Teil liegend aufnehmen können, und um solche, die Platz für das Einsetzen der Nachbarteile lassen |
| Sperrbereich | ein Quader dort, wo das Fixture steht, ist für Greifer und Arm tabu | entfällt, die Baugruppe darf überall stehen |

### 5.3 Ablage statt Fixture (`run_pickup_gen.py`)

Die Teile werden einzeln platziert, in umgekehrter Aufnahme-Reihenfolge. So liegen bei jeder Aufnahme genau die Teile
daneben, die im echten Ablauf dann noch dort liegen. Ein Platz wird nur genommen, wenn

- er auf dem Brett liegt, außerhalb der Montagezone, mit 2 cm Abstand zu den anderen Teilen,
- die Aufnahme dort alle Prüfregeln besteht, mit geöffnetem und mit geschlossenem Greifer,
- der Arm gerade in den Griff hinein und mit dem Teil gerade nach oben herauskommt. Das wird mit der Routine des Planers
  selbst getestet, damit kein Platz gewählt wird, den die Bewegungsplanung später ablehnt.

Für jede Aufnahme werden mehrere Armstellungen berechnet und die am besten konditionierte behalten. Ergebnis sind die
Lagen der Teile, die geprüfte Armstellung je Aufnahme und die Schablonen als PDF.

### 5.4 Strenger Planer für den realen Arm (`ArmMotionPlanner.strict`)

Der freie Pfad kommt wie im Original aus RRT-Connect mit Glättung. Neu ist:

- Jeder Pfad beginnt und endet mit einem geraden Stück: aus einer Loslass- oder Greifstellung gerade heraus, bis alle
  Abstände stimmen, und gerade in die nächste Stellung hinein.
- Ein getragenes Teil wird erst gerade angehoben und bleibt außerhalb der Montagezone 3 cm über dem Tisch.
- Virtuelle Wände, Singularitäten und Gelenkgrenzen sind Bedingungen des Planers. Geplant wird mit einer kleinen Reserve
  über den geprüften Werten.
- Findet der Planer keinen Pfad, sagt er, was die gerade Bewegung gestoppt hat (z. B. „virtuelle Wand nach 0,25 cm").

### 5.5 Prüfregeln (`plan_checks.py`, Werte in `config.py`)

Im Original gibt es nur die Puffer des Planers. Hier wird jede Stellung jedes Pfads des realen Arms nachgemessen:

| Regel | Wert |
|---|---|
| Armglieder zu jedem Teil; Greifer und getragenes Teil zu liegenden Teilen | mindestens 1,0 cm |
| Greifer zum Teil, das er gleich greift | keine Berührung auf der ganzen Anfahrt |
| Greifer und getragenes Teil zur Baugruppe | keine Berührung; im Vorbeifahren mindestens 1,0 cm |
| Greifer zu den eigenen Armgliedern | mindestens 1,0 cm |
| getragenes Teil über dem Tisch, außerhalb der Montagezone | mindestens 3,0 cm |
| getragenes Teil über dem Tisch, in der Montagezone (Baugruppe plus 4 cm Rand) | mindestens 0,2 cm |
| Greifer / Arm über dem Tisch | mindestens 0,2 cm / 1,0 cm |
| virtuelle Wände | mindestens 1,0 cm |
| Singularität (kleinster Singulärwert der Jacobi-Matrix) | mindestens 0,10; beim Einsetzen 0,15 |
| jedes Gelenk zu seinen Grenzen | mindestens 0,5 rad |
| Teil reicht in die Finger | mindestens 0,3 cm |

### 5.6 Schleife mit Ursachen (`run_planning_flat.py`)

Im Original endet ein Lauf, sobald eine Stufe scheitert. Hier schreibt die Stufe auf, was sich ändern muss:

| Ursache | Beispiel | Reaktion |
|---|---|---|
| Griff | ein Einschubpfad bricht eine Regel; kein Ablageplatz auf dem ganzen Brett | der Griff wird verworfen, die nächstbeste Reihenfolge geplant |
| Ablageplatz | keine gerade Anfahrt dort, wo das Teil liegt | der Griff bleibt, das Teil wird woanders abgelegt |
| Einschub selbst | das Teil kommt nicht an seinen Einschub-Startpunkt, egal wie es gehalten wird | die Planung endet mit dieser Meldung |

### 5.7 Platz von Baugruppe und Teilen

Im Original steht die Baugruppe fest bei (0, -15) cm auf dem Brett, das Fixture ab y = 10 cm. In der
Flat-Pipeline sind der Platz der Baugruppe (`--assembly-center`) und der Platz, um den die Teile abgelegt werden
(`--pickup-center`), Eingaben und werden im Log gespeichert. Dazu gehören zwei Werkzeuge:

- `run_workspace_search.py` bewertet Plätze der Baugruppe ohne zu planen und plant am besten.
- `run_position_test.py` lässt die Pipeline an vielen Platzpaaren laufen und berichtet je Paar.

## 6. Was gleich geblieben ist

- Stufe 1 (Zerlegung, Einschubpfade) und Stufe 3 (Reihenfolgen-Baum) sind unverändert.
- Die Bewertung in der Reihenfolgen-Optimierung ist die des Originals.
- Die Einschubpfade stammen weiter aus Griff und Zerlegeplan. Sie werden geprüft, aber nicht neu geplant.
- Der freie Pfad kommt weiter aus RRT-Connect mit Glättung.
- Das Format von `motion.pkl`, das Training der Policy und die Ausführung mit `real_robot/run.py` sind geblieben.
  `run.py` lehnt jetzt Pläne ab, die für eine andere Roboterhöhe gemacht wurden oder die Schlussprüfung nicht bestanden
  haben.

## 7. Dateien

| Datei | Status | Inhalt |
|---|---|---|
| `planning/run_planning_flat.py`, `.sh` | neu | der Ablauf mit Schleife |
| `planning/flat_grasp.py` | neu | Ruhelagen, Aufnahme-Test für liegende Teile, Fingerkontakt, Kriterien, Bewertung und Auswahl der Griffe |
| `planning/run_pickup_gen.py` | neu | Griff-Filter, Reihenfolgen-Optimierung mit Bedingung für das erste Teil, Ablage, Schablonen |
| `planning/plan_checks.py`, `verify_plan.py` | neu | Prüfregeln, Schlussprüfung |
| `planning/run_workspace_search.py`, `run_position_test.py` | neu | Platzsuche, Positions-Test |
| `planning/robot/panda_condition.py` | neu | Gelenkgrenzen und Jacobi-Matrix des Panda |
| `planning/test_plan_rules.py`, `test_flat_grasp.py`, `test_workspace_search.py` | neu | 39 + 42 + 9 Tests |
| `planning/run_grasp_arm_gen.py` | erweitert | Schalter `--flat`; ohne ihn wie im Original |
| `planning/run_motion_plan.py` | erweitert | läuft ohne Fixture; Pfade des realen Arms geprüft und neu geplant; `--single-arm`, `--no-viz`; nennt Griff und Ursache eines Fehlschlags |
| `planning/robot/motion_plan_arm.py` | erweitert | strenger Modus (5.4) |
| `planning/robot/workcell.py`, `util_arm.py`, `geometry.py`, `config.py` | erweitert | Roboterhöhe, Tischhöhe, virtuelle Wände, gespeicherte Plätze, Werte der Regeln |
| `planning/run_preced_plan.py`, `run_seq_plan.py`, `run_seq_opt.py`, `run_fixture_gen.py` | wenige Zeilen | Betrieb ohne Bildschirm, ausführliche Ausgabe als Standard; `run_preced_plan.py` und `run_seq_plan.py` lesen den gespeicherten Platz der Baugruppe |
| `real_robot/run.py`, `robot_interface.py` | erweitert | `--single-arm`, Ablehnung unpassender Pläne, Korrekturen bei der Ausführung |

## 8. Stand und Grenzen

Läufe auf dem Stand vom 12.11.2026 abends (Einzelheiten in `assembly_runs/README.md`):

- Alle zehn Baugruppen am Standardplatz: 8 mit verifiziertem Plan (socket_set_small, socket_set, beam, human_normal, car,
  gamepad, cooling_manifold, duct), 2 ohne (plumbers_block, stool_circular).
- Positions-Test mit 10 Plätzen der Baugruppe und je 3 Plätzen der Teile: beam 30 von 30, gamepad 30 von 30,
  human_normal 30 von 30 verifiziert.

Grenzen, die es im Original so nicht gibt, weil dort das Fixture die Teile hält:

- **Flache, schmale Teile haben keinen Griff.** Die Schrauben von plumbers_block und die Beine von stool_circular liegen
  so niedrig, dass die Finger nicht 3 mm tief anliegen können, ohne dem Tisch näher als 0,5 cm zu kommen.
- **Geneigte Hände.** Die Hand kommt beim Aufnehmen oft schräg, bei einzelnen Teilen über 50°.
- **Teile in wenig stabiler Lage.** car, cooling_manifold und duct haben Teile, die nur in einer Lage mit 11° bis 17°
  Kippwinkel aufgenommen werden können.

Offen:

- Die aufgeblasenen Teil-Netze, gegen die der strenge Planer prüft, haben bei 8 von 62 Teilen lange Spitzen. Der Planer
  lehnt dort freie Stellungen ab. Verifizierte Pläne sind davon nicht betroffen, weil die Prüfung echte Abstände misst.
  Gemessen, noch nicht geändert.
- Drei Eingriffe sind noch nicht bestätigt: der Griff-Test für die geraden Bewegungen an der Baugruppe, die eigene
  Berechnung der Ruhelagen, die Speicherfreigabe im Planer.
- Die Kamerahalterung am Handgelenk und ihr Kabel sind nicht im Robotermodell.

## 9. Bedienung

```bash
conda activate fabrica
bash planning/run_planning.sh      EXP_NAME ASSEMBLY     # Original: zwei Arme, Fixture
bash planning/run_planning_flat.sh EXP_NAME ASSEMBLY     # Flat: ein realer Arm, Teile liegen flach
python planning/run_position_test.py --exp-name EXP_NAME --assembly ASSEMBLY --precedence-from LOG   # viele Plätze
python planning/test_plan_rules.py && python planning/test_flat_grasp.py && python planning/test_workspace_search.py
```

Ergebnis der Flat-Pipeline in `logs/EXP_NAME/ASSEMBLY/`: `motion.pkl` (führt `real_robot/run.py --single-arm` aus),
`fixture/pickup_template_a4.pdf` und `fixture/assembly_template_a4.pdf` (Schablonen, in 100 % drucken),
`verify_report.json`, `workcell.json`.
