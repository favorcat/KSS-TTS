gsap.set('svg', {
    visibility: 'visible'
  })
  const show = () => {
    gsap.to('#bg *', {
      drawSVG: `${100 - easeFunc(tween.ratio) * 50}% ${easeFunc(tween.ratio) * 50}%`,
      stagger: 0.06,
      duration: 0.6
    })
  }
  const easeFunc = gsap.parseEase("slow(0.83, 0.18, true)");
  const tween = gsap.to({val: 0}, {
    val: 1,
    ease: "sine",
    onUpdate: show,
    repeat: -1,
    yoyo: true,
    duration: 0.6,
  });
  
  
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