$(function () {
    $('#predictCaseIssues').bind('click', function () {
        // console.log("Predicting...");
        // console.log($SCRIPT_ROOT);
        $.post($SCRIPT_ROOT + '/predict_case_issues', {
            inputText: $('#inputText').val(),
        }, function (data) {
            // $("#result").text(data.result);
            document.getElementById("allPredictedTags").innerHTML = data;

            var buttons = document.getElementsByClassName("predicted");
            var count = buttons.length;
            while (count--) {
                buttons[count].addEventListener('click', function () {
                    var selected = $('#correctCaseIssues').select2('data');
                    arr = [];
                    for (let i = 0; i < selected.length; i++) {
                        arr.push(selected[i].id);
                    }
                    if (arr.indexOf(this.value) < 0) {
                        arr.push(this.value);
                        $('#correctCaseIssues').val(arr);
                        $('#correctCaseIssues').trigger('change');
                    }
                });
                buttons[count].addEventListener('mouseover', function () {
                    // console.log("Hovering over " + this.value);
                    $("#infoText").height($("#textSlide")[0].scrollHeight);
                    $("#infoSlide").height($("#textSlide")[0].scrollHeight);
                    console.log(this.getAttribute("data-info"))
                    $("#infoText").html(this.getAttribute("data-info"));
                    $('#myCarousel').carousel(1);
                    $('.carousel').carousel({
                        "interval": false,
                        "data-interval": false,
                        "data-bs-interval": false
                    });
                });
                buttons[count].addEventListener('mouseleave', function () {
                    // console.log("Leaving " + this.value);
                    $('#myCarousel').carousel(0)
                    $('.carousel').carousel({
                        "interval": false,
                        "data-interval": false,
                        "data-bs-interval": false
                    });
                });
            }
        });
        return false;
    });
});