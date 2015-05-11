<?php
if(isset($_GET['put']))
{
	$ob = new stdClass();
	$ob->name = $_GET['put'];
	$ob->value = $_GET['value'];
	$ob->date = $_GET['date'];
	$js = json_encode($ob);
	$filename = "escrow_" . $ob->name . ".json";
	$timenow = time();
	if(file_put_contents($filename, $js) === FALSE)
		print("Failed to create file " . $filename);
	else
		print("Not until " . $ob->date . ". (Currently " . $timenow . ". Seconds remaining=" . ($ob->date - $timenow) . ".)");
}
else if(isset($_GET['get']))
{
	$filename = "escrow_" . $_GET['get'] . ".json";
	$js = file_get_contents($filename);
	if($js === FALSE)
		print("File not found: " . $filename);
	else
	{
		$ob = json_decode($js);
		$timenow = time();
		if($timenow < $ob->date)
			print("Not until " . $ob->date . ". (Currently " . $timenow . ". Seconds remaining=" . ($ob->date - $timenow) . ".)");
		else
			print($ob->value);
	}
}
else
{
	print("<h3>Put</h3>\n");
	print("<form method=\"get\">\n");
	print("<input type=\"text\" name=\"put\" value=\"test\">\n");
	print("<input type=\"text\" name=\"value\" value=\"secret\">\n");
	print("<input type=\"text\" name=\"date\" value=\"" . time() . "\">\n");
	print("<input type=\"submit\" name=\"Submit\" value=\"Put\">\n");
	print("</form><br><br><br><br>\n");

	print("<h3>Get</h3>\n");
	print("<form method=\"get\">\n");
	print("<input type=\"text\" name=\"get\" value=\"test\">\n");
	print("<input type=\"submit\" name=\"Submit\" value=\"Get\">\n");
	print("</form>\n");
}
?>
