@echo off
echo Paste the path to Java installation folder and press enter.
echo(
echo    Example how the path should look:
echo(
echo        C:\Program Files\Java\jdk-XX.YY.ZZ
echo(
echo    where XX, YY, and ZZ refer to the version numbers for the Java
echo    version you installed. DO NOT enter XX, YY, ZZ; locate the path
echo    under "C:\Program Files\Java\", and copy and paste the path here.
echo    If you cannot find the Java folder under "Program Files", it is
echo    possible that Java is installed under "Program Files (x86)".
echo(
set /p "java_path=>"
setx -m JAVA_HOME "%java_path:"=%"
setx -m PATH "%PATH%;%java_path:"=%\bin";