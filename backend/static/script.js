//  Element
var inputarea = document.getElementById("input");
var outputarea = document.getElementById("output");
var cousor = document.getElementById("cousor");


output.addEventListener("click", function (event) {
  inputarea.style.zIndex = 4;
  inputarea.style.color = "black";
}, false);

inputarea.addEventListener("input", function (event) {
  outputarea.innerHTML = inputarea.value;
  inputarea.style.zIndex = 4;
  //
  refreshHighlighting();
}, false);


// Refresh highlighting when blured focus from textarea.
inputarea.addEventListener("blur", function (event) {
  refreshHighlighting();
}, false)


function refreshHighlighting() {
  hljs.highlightBlock(outputarea);
  //setTimeout("refreshHighlighting()", 1000);
  inputarea.style.zIndex = 0;
  inputarea.style.color = "transparent";
}
refreshHighlighting();