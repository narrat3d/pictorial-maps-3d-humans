/*global define*/
define([
    '../SpeechBubble/HtmlBillboardCollection',
], function(
    HtmlBillboardCollection) {
    'use strict';

    var SpeechBubbleLayer = function (viewer) {
        document.body.insertAdjacentHTML(
            "beforeend",
            `<template id="speech-bubble-template">
			<div class="speech-bubble">
				<div class="text"></div>
			</div>
		</template>`
        );

        // eslint-disable-next-line no-undef
        const htmlLayer = new HtmlBillboardCollection(viewer.scene);

        this.add = function (text, position) {
            const htmlLayerElem =
                document.getElementsByClassName("cesium-html-layer")[0];
            const speechBubbleTemplate = document.getElementById(
                "speech-bubble-template"
            );
            const speechBubbleTemplateElem = document.importNode(
                speechBubbleTemplate.content,
                true
            );

            const textElem = speechBubbleTemplateElem.querySelector(".text");
            textElem.innerText = text;

            htmlLayerElem.appendChild(speechBubbleTemplateElem);
            const speechBubbleElements =
                htmlLayerElem.getElementsByClassName("speech-bubble");
            const speechBubbleElem =
                speechBubbleElements[speechBubbleElements.length - 1];

            const htmlElem = htmlLayer.add();
            htmlElem.element = speechBubbleElem;
            htmlElem.position = position;

            //by having the speech bubble attached to the DOM, we can get its width and height
            const height = speechBubbleElem.clientHeight;
            const anchorHeight = parseInt(
                getComputedStyle(speechBubbleElem, ":after").borderTopWidth
            );
            const width = speechBubbleElem.clientWidth;

            htmlElem.offsetLeft = -Math.floor(width / 2);
            htmlElem.offsetTop = -height - anchorHeight;

            speechBubbleElem.style.position = "absolute";
            speechBubbleElem.style.visibility = "visible";

            htmlElem.setText = function (text) {
                textElem.innerText = text;

                //update anchor position as height may have changed
                const newHeight = speechBubbleElem.clientHeight;
                htmlElem.offsetTop = -newHeight - anchorHeight;
            };

            htmlElem.setPosition = function (position) {
                htmlElem.position = position;
            };

            return htmlElem;
        };

        this.remove = function (htmlElem) {
            htmlLayer.remove(htmlElem);
        };
    };

    return SpeechBubbleLayer;
});
