function toggleMenu(){
  const menu = document.getElementById("mobileMenu");
  const isOpen = menu.style.display === "block";
  menu.style.display = isOpen ? "none" : "block";
  menu.setAttribute("aria-hidden", isOpen ? "true" : "false");
}
function closeMenu(){
  const menu = document.getElementById("mobileMenu");
  menu.style.display = "none";
  menu.setAttribute("aria-hidden", "true");
}