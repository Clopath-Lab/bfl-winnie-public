$('.pagination').bootpag({
    total: 5
}).on("page", function (event, num) {
    $(".content").html("Page " + num); // or some ajax content loading...

    // ... after content load -> change total to 10
    $(this).bootpag({ total: 10, maxVisible: 10 });

});
